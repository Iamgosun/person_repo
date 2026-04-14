from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from bayesvlm.adapter import ADAPTER_REGISTRY, build_adapter
from bayesvlm.text_priors import _build_templates
from bayesvlm.utils import load_model


class VLMAdapter(nn.Module):
    """
    CLIP/SigLIP adapter 封装。

    设计目标
    --------
    1. image/text encoder 与 VLM 标量参数保持冻结；
    2. adapter 接收类别文本 prototype 作为先验；
    3. 同时兼容 raw-image loader 与 cached-feature loader；
    4. 不再把“deterministic backbone”错误地收窄成
       “adapter 默认也必须是纯确定性前向”。

    兼容性
    ------
    - 保留 ``text_covariance`` 入参，避免旧调用报错；
    - 既支持外部传入 ``image_encoder/text_encoder/vlm``，
      也支持根据 cfg 自动调用 ``load_model``；
    - forward / zero_shot_logits 新增支持：
        * image_features 显式传入
        * batch["image_embeds"]
        * batch["embeds"]
      从而兼容图像特征缓存后的训练/评估流水线。
    """

    def __init__(
        self,
        cfg: Any,
        classnames: Sequence[str],
        text_covariance=None,
        image_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        vlm: nn.Module | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.classnames = [str(x) for x in classnames]
        self.text_covariance = text_covariance

        self.adapter_name = str(self._cfg_get("adapter_name", "LP")).upper()
        self.initialization = str(self._cfg_get("initialization", "MEAN"))
        self.dataset_name = str(self._cfg_get("datasetname", self._cfg_get("dataset", "")))
        self.train_logit_scale = bool(self._cfg_get("train_logit_scale", False))

        if self.adapter_name not in ADAPTER_REGISTRY:
            raise ValueError(
                f"Unsupported adapter_name: {self.adapter_name}. "
                f"Available adapters: {sorted(ADAPTER_REGISTRY.keys())}"
            )

        self.image_encoder, self.text_encoder, self.vlm = self._resolve_backbones(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            vlm=vlm,
        )

        self._freeze_backbones()

        base_text_features = self._build_base_text_features()
        self.register_buffer("base_text_features", base_text_features)

        self.adapter_cls = ADAPTER_REGISTRY[self.adapter_name]
        self.adapter_input_kind = getattr(self.adapter_cls, "input_kind", "vector")
        self.adapter = self._build_adapter()

        # 兼容旧代码
        self.logit_scale = self.vlm.logit_scale
        self.logit_bias = getattr(self.vlm, "logit_bias", None)

    def _cfg_tensor(self, key: str) -> torch.Tensor | None:
        value = self._cfg_get(key, None)
        if value is None:
            return None

        if torch.is_tensor(value):
            return value.to(
                device=self.base_text_features.device,
                dtype=self.base_text_features.dtype,
            )

        return torch.as_tensor(
            value,
            device=self.base_text_features.device,
            dtype=self.base_text_features.dtype,
        )

    def _cfg_get(self, key: str, default=None):
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def _resolve_backbones(
        self,
        image_encoder: nn.Module | None,
        text_encoder: nn.Module | None,
        vlm: nn.Module | None,
    ):
        if image_encoder is not None and text_encoder is not None and vlm is not None:
            return image_encoder, text_encoder, vlm

        model_str = str(self._cfg_get("model", self._cfg_get("model_str", "clip-base")))
        device = str(self._cfg_get("device", "cpu"))
        local_model_path = self._cfg_get(
            "local_model_path",
            self._cfg_get("model_name_or_path", None),
        )

        image_encoder, text_encoder, vlm = load_model(
            model_str=model_str,
            device=device,
            local_model_path=local_model_path,
        )
        return image_encoder, text_encoder, vlm

    def _freeze_backbones(self):
        if hasattr(self.image_encoder, "freeze_all_layers"):
            self.image_encoder.freeze_all_layers()
        else:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if hasattr(self.text_encoder, "freeze_all_layers"):
            self.text_encoder.freeze_all_layers()
        else:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        for p in self.vlm.parameters():
            p.requires_grad = False

        if hasattr(self.vlm, "logit_scale") and self.vlm.logit_scale is not None:
            self.vlm.logit_scale.requires_grad = self.train_logit_scale
        if hasattr(self.vlm, "logit_bias") and self.vlm.logit_bias is not None:
            self.vlm.logit_bias.requires_grad = False

        self.image_encoder.eval()
        self.text_encoder.eval()
        self.vlm.eval()

    def train(self, mode: bool = True):
        """
        只让 adapter 切换 train/eval；
        frozen backbone 始终保持 eval。
        """
        super().train(mode)
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.vlm.eval()
        self.adapter.train(mode)
        return self

    def _runtime_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.image_encoder.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    def _image_encoder_dtype(self) -> torch.dtype:
        try:
            return next(self.image_encoder.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _text_encoder_device(self) -> torch.device:
        try:
            return next(self.text_encoder.parameters()).device
        except StopIteration:
            return self._runtime_device()

    @torch.no_grad()
    def _build_base_text_features(self) -> torch.Tensor:
        templates = _build_templates(self.dataset_name)
        device = self._text_encoder_device()
        class_features = []

        self.text_encoder.eval()
        for class_name in self.classnames:
            prompts = [template.format(class_name.replace("_", " ")) for template in templates]
            text_embeds = self.text_encoder(prompts)
            if hasattr(text_embeds, "embeds"):
                text_embeds = text_embeds.embeds
            if isinstance(text_embeds, tuple):
                text_embeds = text_embeds[0]
            text_embeds = text_embeds.to(device=device, dtype=torch.float32)
            class_features.append(text_embeds.mean(dim=0))

        return torch.stack(class_features, dim=0)

    def _adapter_kwargs(self) -> dict:
        kwargs = {}
        name = self.adapter_name

        if name == "TR":
            kwargs["alpha"] = float(self._cfg_get("taskres_alpha", 0.5))

        elif name == "CLIPA":
            kwargs["ratio"] = float(self._cfg_get("clipa_ratio", 0.2))
            hidden_dim = self._cfg_get("clipa_hidden_dim", None)
            if hidden_dim is not None:
                hidden_dim = int(hidden_dim)
                if hidden_dim > 0:
                    kwargs["hidden_dim"] = hidden_dim

        elif name == "TIPA":
            kwargs["alpha"] = float(self._cfg_get("tipa_alpha", 1.0))
            kwargs["beta"] = float(self._cfg_get("tipa_beta", 1.0))

        elif name == "GAUSSIAN_PER_CLASS":
            kwargs["prior_sigma"] = float(self._cfg_get("gaussian_prior_sigma", 0.01))
            kwargs["mc_samples"] = int(self._cfg_get("gaussian_mc_samples", 3))
            kwargs["anneal_start_epoch"] = int(
                self._cfg_get("gaussian_anneal_start_epoch", 20)
            )
            kwargs["total_epochs"] = int(
                self._cfg_get("epochs", self._cfg_get("max_epoch", 300))
            )

        elif name in {"BAYESADAPTER", "BAYES_ADAPTER"}:
            kwargs["prior_sigma"] = float(
                self._cfg_get("bayesadapter_prior_sigma", 0.01)
            )
            kwargs["train_mc_samples"] = int(
                self._cfg_get("bayesadapter_train_mc_samples", 3)
            )
            kwargs["eval_mc_samples"] = int(
                self._cfg_get("bayesadapter_eval_mc_samples", 10)
            )
            kwargs["kl_scale_divisor"] = float(
                self._cfg_get("bayesadapter_kl_scale_divisor", 1000.0)
            )
            kwargs["total_epochs"] = int(
                self._cfg_get("epochs", self._cfg_get("max_epoch", 300))
            )
            kwargs["covariance_mode"] = str(
                self._cfg_get("bayesadapter_covariance_mode", "paper_scalar")
            ).lower()

            prior_mu = self._cfg_tensor("bayesadapter_prior_mu")
            prior_log_sigma = self._cfg_tensor("bayesadapter_prior_log_sigma")

            if prior_mu is not None:
                kwargs["prior_mu"] = prior_mu

            if prior_log_sigma is not None:
                kwargs["prior_log_sigma"] = prior_log_sigma

        elif name == "VMFPROTO":
            posterior_eta = self._cfg_tensor("vmfproto_posterior_eta")
            A_img_inv = self._cfg_tensor("vmfproto_A_img_inv")
            B_img_inv = self._cfg_tensor("vmfproto_B_img_inv")

            if posterior_eta is None or A_img_inv is None or B_img_inv is None:
                raise ValueError("VMFPROTO requires vmfproto_posterior_eta / A_img_inv / B_img_inv in cfg")

            kwargs["posterior_eta"] = posterior_eta
            kwargs["A_img_inv"] = A_img_inv
            kwargs["B_img_inv"] = B_img_inv
            kwargs["kappa_scale"] = float(self._cfg_get("vmf_kappa_scale", 1.0))
            kwargs["eps"] = float(self._cfg_get("vmf_eps", 1e-6))
            kwargs["kappa_max"] = float(self._cfg_get("vmf_kappa_max", 500.0))

        return kwargs

    def _build_adapter(self) -> nn.Module:
        if self.adapter_input_kind != "vector":
            raise ValueError(
                f"Current VLMAdapter only supports vector adapters, "
                f"got {self.adapter_input_kind}."
            )
        return build_adapter(
            adapter_name=self.adapter_name,
            base_text_features=self.base_text_features,
            initialization=self.initialization,
            **self._adapter_kwargs(),
        )

    def trainable_parameters(self):
        params = list(self.adapter.parameters())
        if (
            self.train_logit_scale
            and hasattr(self.vlm, "logit_scale")
            and self.vlm.logit_scale is not None
        ):
            params.append(self.vlm.logit_scale)
        return params

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.adapter, "set_epoch"):
            self.adapter.set_epoch(epoch)

    def adapter_regularization_loss(self):
        if hasattr(self.adapter, "regularization_loss"):
            return self.adapter.regularization_loss()
        zero = torch.zeros((), device=self._runtime_device(), dtype=torch.float32)
        return zero, {}

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(
            device=self._runtime_device(),
            dtype=self._image_encoder_dtype(),
        )
        image_features = self.image_encoder(image)
        if hasattr(image_features, "embeds"):
            image_features = image_features.embeds
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        return image_features

    def _coerce_feature_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return features.to(device=self._runtime_device(), dtype=torch.float32)

    def _extract_image_state(
        self,
        batch=None,
        image: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
    ):
        """
        返回:
            {
                "image_features": [B, D],
                "activations":    [B, H] or None,
                "residuals":      [B, D] or None,
            }
        """
        if image_features is not None:
            return {
                "image_features": self._coerce_feature_tensor(image_features),
                "activations": None,
                "residuals": None,
            }

        if isinstance(batch, dict):
            if "image_embeds" in batch:
                activations = batch.get("activations", None)
                residuals = batch.get("residuals", None)
                return {
                    "image_features": self._coerce_feature_tensor(batch["image_embeds"]),
                    "activations": None if activations is None else self._coerce_feature_tensor(activations),
                    "residuals": None if residuals is None else self._coerce_feature_tensor(residuals),
                }
            if "embeds" in batch:
                activations = batch.get("activations", None)
                residuals = batch.get("residuals", None)
                return {
                    "image_features": self._coerce_feature_tensor(batch["embeds"]),
                    "activations": None if activations is None else self._coerce_feature_tensor(activations),
                    "residuals": None if residuals is None else self._coerce_feature_tensor(residuals),
                }

        if image is None:
            if batch is None:
                raise ValueError("batch、image、image_features 不能同时为空")
            image = batch["image"] if isinstance(batch, dict) else batch

        encoded = self.image_encoder(image, return_activations=True)
        return {
            "image_features": self._coerce_feature_tensor(encoded.embeds),
            "activations": self._coerce_feature_tensor(encoded.activations),
            "residuals": self._coerce_feature_tensor(encoded.residuals),
        }

    def _extract_image_features(
        self,
        batch=None,
        image: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        优先级：
        1) 显式传入 image_features
        2) batch["image_embeds"]
        3) batch["embeds"]
        4) 显式传入 image
        5) batch["image"]
        """
        if image_features is not None:
            return self._coerce_feature_tensor(image_features)

        if isinstance(batch, dict):
            if "image_embeds" in batch:
                return self._coerce_feature_tensor(batch["image_embeds"])
            if "embeds" in batch:
                return self._coerce_feature_tensor(batch["embeds"])

        if image is None:
            if batch is None:
                raise ValueError("batch、image、image_features 不能同时为空")
            image = batch["image"] if isinstance(batch, dict) else batch

        return self._encode_image(image)

    def _apply_adapter(
        self,
        image_features: torch.Tensor,
        activations: torch.Tensor | None = None,
        residuals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(self.adapter, "forward_with_aux"):
            return self.adapter.forward_with_aux(
                image_features=image_features,
                activations=activations,
                residuals=residuals,
                logit_scale=self.vlm.logit_scale,
            )

        logits = self.adapter(image_features, self.vlm.logit_scale)
        if getattr(self.vlm, "logit_bias", None) is not None:
            logits = logits + self.vlm.logit_bias.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        return logits

    def forward(
        self,
        batch=None,
        image: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        state = self._extract_image_state(
            batch=batch,
            image=image,
            image_features=image_features,
        )
        feats = state["image_features"]
        logits = self._apply_adapter(
            image_features=feats,
            activations=state["activations"],
            residuals=state["residuals"],
        )

        if return_features:
            return logits, feats
        return logits

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        features = self._coerce_feature_tensor(features)
        return self._apply_adapter(features)

    @torch.no_grad()
    def zero_shot_logits(
        self,
        batch=None,
        image: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self._extract_image_features(
            batch=batch,
            image=image,
            image_features=image_features,
        )
        base_text_features = self.base_text_features.to(
            device=feats.device,
            dtype=feats.dtype,
        )
        logits = self.vlm(feats, base_text_features)

        if getattr(self.vlm, "logit_bias", None) is not None:
            logits = logits + self.vlm.logit_bias.to(
                device=logits.device,
                dtype=logits.dtype,
            )
        return logits