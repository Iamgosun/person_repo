from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from bayesvlm.adapter import ADAPTER_REGISTRY, build_adapter
from bayesvlm.text_priors import _build_templates
from bayesvlm.utils import load_model


class VLMAdapter(nn.Module):
    """
    确定性 CLIP/SigLIP adapter 封装。

    设计目标：
    1. 与 few-shot CoOp 保持同口径：冻结 image/text encoder 与 VLM 标量参数；
    2. adapter 只接收确定性的类别文本 prototype；
    3. forward 输出普通 logits，方便与 few-shot CoOp 做公平对比。

    兼容性说明：
    - 保留 ``text_covariance`` 入参，避免旧调用直接报错，但该参数在确定性版本里不会被使用。
    - 既支持外部传入 ``image_encoder/text_encoder/vlm``，也支持根据 cfg 自动调用 ``load_model``。
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

        # 兼容旧代码，保留直接访问 logit_scale/logit_bias 的方式。
        self.logit_scale = self.vlm.logit_scale
        self.logit_bias = getattr(self.vlm, "logit_bias", None)

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
        local_model_path = self._cfg_get("local_model_path", self._cfg_get("model_name_or_path", None))

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
            self.vlm.logit_scale.requires_grad = False
        if hasattr(self.vlm, "logit_bias") and self.vlm.logit_bias is not None:
            self.vlm.logit_bias.requires_grad = False

        self.image_encoder.eval()
        self.text_encoder.eval()
        self.vlm.eval()

    def train(self, mode: bool = True):
        """
        只让 adapter 切换 train/eval；
        frozen 的 image/text/vlm backbone 始终保持 eval 模式。
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

    def _build_adapter(self) -> nn.Module:
        if self.adapter_input_kind != "vector":
            raise ValueError(
                f"Current deterministic VLMAdapter only supports vector adapters, got {self.adapter_input_kind}."
            )
        return build_adapter(
            adapter_name=self.adapter_name,
            base_text_features=self.base_text_features,
            initialization=self.initialization,
        )

    def trainable_parameters(self):
        return self.adapter.parameters()

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

    def _apply_adapter(self, image_features: torch.Tensor) -> torch.Tensor:
        logits = self.adapter(image_features, self.vlm.logit_scale)
        if getattr(self.vlm, "logit_bias", None) is not None:
            logits = logits + self.vlm.logit_bias.to(device=logits.device, dtype=logits.dtype)
        return logits

    def forward(
        self,
        batch=None,
        image: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        if image is None:
            if batch is None:
                raise ValueError("batch 和 image 不能同时为空")
            image = batch["image"] if isinstance(batch, dict) else batch

        image_features = self._encode_image(image)
        logits = self._apply_adapter(image_features)

        if return_features:
            return logits, image_features
        return logits

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(device=self._runtime_device(), dtype=torch.float32)
        return self._apply_adapter(features)

    @torch.no_grad()
    def zero_shot_logits(self, batch=None, image: torch.Tensor | None = None) -> torch.Tensor:
        if image is None:
            if batch is None:
                raise ValueError("batch 和 image 不能同时为空")
            image = batch["image"] if isinstance(batch, dict) else batch

        image_features = self._encode_image(image)
        base_text_features = self.base_text_features.to(
            device=image_features.device,
            dtype=image_features.dtype,
        )
        logits = self.vlm(image_features, base_text_features)
        if getattr(self.vlm, "logit_bias", None) is not None:
            logits = logits + self.vlm.logit_bias.to(device=logits.device, dtype=logits.dtype)
        return logits