from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 中文说明：
# 这里优先使用 scipy.special.ive 来稳定计算 vMF 的 log normalizer。
# 如果环境里没有 scipy，则退回到原先的近似实现，但会打印 backend=fallback_asymptotic。
try:
    import numpy as np
    from scipy.special import ive as scipy_ive
except Exception:  # pragma: no cover
    np = None
    scipy_ive = None


TextStateKind = Literal["vector", "distribution"]


class AdapterMethod(nn.Module):
    """
    Base class for CLIP/SigLIP adapters。

    说明
    ----
    - image/text encoder 可以是冻结的、确定性的；
    - 但 adapter 本身不要求一定是“确定性参数化”。
    - 因此像 GaussianPerClass 这类 adapter，forward 可以按采样原型后
      MC 平均 logits 的方式工作，而不是被强行降成 posterior mean 的纯确定性分类头。
    """

    input_kind: TextStateKind = "vector"

    def __init__(self, initialization: str = "MEAN"):
        super().__init__()
        self.initialization = str(initialization)

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def regularization_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        adapter 自身的附加正则。
        默认无正则。
        """
        device = self._runtime_device()
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {}

    def set_epoch(self, epoch: int) -> None:
        """
        给需要 epoch-schedule 的 adapter（如 Gaussian KL annealing）用。
        默认什么也不做。
        """
        return None

    def _runtime_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    @staticmethod
    def _normalize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _exp_scale(
        logit_scale: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return logit_scale.exp().to(device=device, dtype=dtype)


class LinearProbeAdapter(AdapterMethod):
    """Standard linear probe over class prototypes."""

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "MEAN",
    ):
        super().__init__(initialization)
        self.prototypes = nn.Parameter(self._init_prototypes(base_text_features))

    def _init_prototypes(self, base_text_features: torch.Tensor) -> torch.Tensor:
        init = self.initialization.upper()

        if init == "RANDOM":
            print(">> Using RANDOM initialization in LinearProbeAdapter")
            weight = torch.empty_like(base_text_features)
            nn.init.kaiming_normal_(weight)
            return weight

        if init in {"MEAN", "ZS", "CROSSMODAL"}:
            print(f">> Using {init} initialization in LinearProbeAdapter")
            return base_text_features.clone()

        print(
            f">> Unrecognized initialization '{self.initialization}', "
            f"fallback to MEAN in LinearProbeAdapter"
        )
        return base_text_features.clone()

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self._normalize_features(image_features)
        prototypes = self._normalize_features(self.prototypes)
        scale = self._exp_scale(logit_scale, image_features.dtype, image_features.device)
        prototypes = prototypes.to(device=image_features.device, dtype=image_features.dtype)
        return image_features @ prototypes.t() * scale


class CrossModalAdapter(LinearProbeAdapter):
    """
    Same parameterization as LP。

    CrossModal 的关键差异主要在 trainer 侧：
    额外采样文本 prototype 作为监督，不在 head 结构本身。
    """

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "CrossModal",
    ):
        super().__init__(base_text_features=base_text_features, initialization=initialization)


class TaskResidualAdapter(AdapterMethod):
    """
    TaskRes-style residual adapter。

    logits(x) = sim(x, base_text + alpha * residual)
    """

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "TR",
        alpha: float = 0.5,
    ):
        super().__init__(initialization)
        self.register_buffer("base_text_features", base_text_features.clone())
        self.alpha = float(alpha)
        self.prototypes = nn.Parameter(torch.zeros_like(base_text_features))

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self._normalize_features(image_features)
        prototypes = self.base_text_features + self.alpha * self.prototypes
        prototypes = self._normalize_features(prototypes)
        prototypes = prototypes.to(device=image_features.device, dtype=image_features.dtype)
        scale = self._exp_scale(logit_scale, image_features.dtype, image_features.device)
        return image_features @ prototypes.t() * scale


class ClipAdapter(AdapterMethod):
    """CLIP-Adapter style residual MLP on image features."""

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "ClipA",
        ratio: float = 0.2,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__(initialization)
        feat_dim = int(base_text_features.shape[-1])
        hidden_dim = int(hidden_dim or max(1, feat_dim // 4))

        self.ratio = float(ratio)
        self.register_buffer("base_text_features", base_text_features.clone())
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_text_features.to(
            device=image_features.device,
            dtype=image_features.dtype,
        )
        prototypes = self._normalize_features(base)

        mlp_dtype = next(self.mlp.parameters()).dtype
        mlp_device = next(self.mlp.parameters()).device
        features_for_mlp = image_features.to(device=mlp_device, dtype=mlp_dtype)

        adapted = self.mlp(features_for_mlp)
        residual = image_features.to(device=adapted.device, dtype=adapted.dtype)
        mixed = self.ratio * adapted + (1.0 - self.ratio) * residual
        mixed = mixed.to(device=image_features.device, dtype=image_features.dtype)
        mixed = self._normalize_features(mixed)

        scale = self._exp_scale(logit_scale, mixed.dtype, mixed.device)
        return mixed @ prototypes.t() * scale


class TipAdapter(AdapterMethod):
    """Tip-Adapter / Tip-Adapter-F style cache adapter."""

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "TipA",
        beta: float = 1.0,
        alpha: float = 1.0,
    ):
        super().__init__(initialization)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.register_buffer("base_text_features", base_text_features.clone())
        self.cache_keys: Optional[nn.Parameter] = None
        self.cache_values: Optional[nn.Parameter] = None

    @torch.no_grad()
    def init_tipadapter(
        self,
        features_train: torch.Tensor,
        labels_train: torch.Tensor,
    ) -> None:
        features_train = torch.as_tensor(
            features_train,
            device=self.base_text_features.device,
            dtype=self.base_text_features.dtype,
        )
        labels_train = torch.as_tensor(
            labels_train,
            device=self.base_text_features.device,
            dtype=torch.long,
        )
        one_hot = F.one_hot(
            labels_train,
            num_classes=self.base_text_features.shape[0],
        ).to(dtype=self.base_text_features.dtype)

        self.cache_keys = nn.Parameter(features_train.clone(), requires_grad=True)
        self.cache_values = nn.Parameter(one_hot.clone(), requires_grad=False)

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_features_norm = self._normalize_features(image_features)

        base = self.base_text_features.to(
            device=image_features.device,
            dtype=image_features.dtype,
        )
        prototypes = self._normalize_features(base)
        scale = self._exp_scale(logit_scale, image_features.dtype, image_features.device)
        logits = image_features_norm @ prototypes.t() * scale

        if self.cache_keys is not None and self.cache_values is not None:
            cache_keys = self._normalize_features(
                self.cache_keys.to(device=image_features.device, dtype=image_features.dtype)
            )
            cache_values = self.cache_values.to(
                device=image_features.device,
                dtype=image_features.dtype,
            )
            affinity = image_features @ cache_keys.t()
            cache_logits = torch.exp((-1.0) * (self.beta - self.beta * affinity)) @ cache_values
            logits = logits + self.alpha * cache_logits.to(
                device=logits.device,
                dtype=logits.dtype,
            )

        return logits


class GaussianPerClassAdapter(AdapterMethod):
    """
    Per-class Gaussian adapter。

    这里不再把“冻结 deterministic CLIP backbone”误解成
    “adapter 也必须默认走 posterior mean 的纯确定性前向”。

    参照你上传的训练器逻辑：
    - 保留 variational_mu / variational_log_sigma / KL
    - forward 默认使用多次 prototype sampling 后的 mean logits
    - KL 权重采用线性 annealing
    """

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "GAUSSIAN_PER_CLASS",
        prior_sigma: float = 0.01,
        mc_samples: int = 3,
        anneal_start_epoch: int = 20,
        total_epochs: int = 300,
    ):
        super().__init__(initialization)
        n_classes, feat_dim = base_text_features.shape
        dtype = base_text_features.dtype
        device = base_text_features.device

        prior_sigma = float(prior_sigma)
        mc_samples = int(mc_samples)
        anneal_start_epoch = int(anneal_start_epoch)
        total_epochs = int(total_epochs)

        self.variational_mu = nn.Parameter(base_text_features.clone())
        self.variational_log_sigma = nn.Parameter(
            torch.full(
                (n_classes, feat_dim),
                math.log(max(prior_sigma, 1e-8)),
                device=device,
                dtype=dtype,
            )
        )

        self.register_buffer("prior_mu", base_text_features.clone())
        self.register_buffer(
            "prior_sigma",
            torch.full((n_classes, feat_dim), prior_sigma, device=device, dtype=dtype),
        )

        self.mc_samples = max(mc_samples, 1)
        self.anneal_start_epoch = max(anneal_start_epoch, 0)
        self.total_epochs = max(total_epochs, 1)
        self.kl_weight = 0.0

    def set_epoch(self, epoch: int) -> None:
        """
        线性 KL annealing。
        epoch 采用 1-based。
        """
        epoch = int(epoch)

        if epoch <= self.anneal_start_epoch:
            self.kl_weight = 0.0
            return

        denom = max(1, self.total_epochs - self.anneal_start_epoch)
        progress = float(epoch - self.anneal_start_epoch) / float(denom)
        self.kl_weight = float(max(0.0, min(progress, 1.0)))

    def sample_prototypes(self, n_samples: Optional[int] = None) -> torch.Tensor:
        n_samples = int(n_samples or self.mc_samples)
        eps = torch.randn(
            (n_samples, *self.variational_mu.shape),
            device=self.variational_mu.device,
            dtype=self.variational_mu.dtype,
        )
        sigma = torch.exp(self.variational_log_sigma)
        return self.variational_mu.unsqueeze(0) + eps * sigma.unsqueeze(0)

    def kl_divergence(self) -> torch.Tensor:
        posterior_sigma = torch.exp(self.variational_log_sigma) + 1e-8
        prior_sigma = self.prior_sigma.to(self.variational_mu.device) + 1e-8
        prior_mu = self.prior_mu.to(self.variational_mu.device)

        kl_per_dim = (
            (posterior_sigma.pow(2) + (self.variational_mu - prior_mu).pow(2))
            / (2 * prior_sigma.pow(2))
            - 0.5
            - torch.log(posterior_sigma)
            + torch.log(prior_sigma)
        )
        return torch.clamp(kl_per_dim.sum(), max=1e4)

    def mc_logits(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> torch.Tensor:
        n_samples = int(
            n_samples
            or (self.train_mc_samples if self.training else self.eval_mc_samples)
        )

        image_features_norm = self._normalize_features(image_features)

        prototypes = self.sample_prototypes(n_samples=n_samples)   # [S, C, D]
        prototypes = self._normalize_features(prototypes)
        prototypes = prototypes.to(
            device=image_features_norm.device,
            dtype=image_features_norm.dtype,
        )

        scale = self._exp_scale(
            logit_scale,
            image_features_norm.dtype,
            image_features_norm.device,
        )

        # official semantics: [S, B, C]
        logits = torch.einsum("bd,scd->sbc", image_features_norm, prototypes) * scale
        return logits

    def forward(self, image_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        return self.mc_logits(
            image_features=image_features,
            logit_scale=logit_scale,
            n_samples=self.train_mc_samples if self.training else self.eval_mc_samples,
        )

    def regularization_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        kl = self.kl_divergence()
        loss = kl * float(self.kl_weight)
        return loss, {
            "loss_kl_raw": float(kl.detach().item()),
            "kl_weight": float(self.kl_weight),
            "loss_kl": float(loss.detach().item()),
        }


class BayesPaperAdapter(AdapterMethod):
    """
    BayesAdapter。

    支持两种 covariance mode:
    1) paper_scalar:
       与论文/当前实现一致。每类一个标量 sigma，协方差为 sigma_c^2 I_D。
    2) diag:
       扩展版。每类一个 D 维对角 sigma，协方差为 diag(sigma_{c,1}^2, ..., sigma_{c,D}^2)。

    默认仍为 paper_scalar，因此不传新参数时保持原逻辑不变。
    """

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "BAYESADAPTER",
        prior_sigma: float = 0.01,
        train_mc_samples: int = 3,
        eval_mc_samples: int = 10,
        total_epochs: int = 300,
        kl_scale_divisor: float = 1000.0,
        covariance_mode: str = "paper_scalar",
        prior_mu: Optional[torch.Tensor] = None,
        prior_log_sigma: Optional[torch.Tensor] = None,
    ):
        super().__init__(initialization)

        n_classes, feat_dim = base_text_features.shape
        dtype = base_text_features.dtype
        device = base_text_features.device

        prior_sigma = float(max(prior_sigma, 1e-8))
        train_mc_samples = int(max(train_mc_samples, 1))
        eval_mc_samples = int(max(eval_mc_samples, 1))
        total_epochs = int(max(total_epochs, 1))
        kl_scale_divisor = float(max(kl_scale_divisor, 1e-8))

        covariance_mode = str(covariance_mode).lower()
        if covariance_mode not in {"paper_scalar", "diag"}:
            raise ValueError(
                f"Unsupported covariance_mode={covariance_mode}. "
                "Choices: ['paper_scalar', 'diag']"
            )
        self.covariance_mode = covariance_mode

        # 先验均值
        if prior_mu is None:
            prior_mu_tensor = base_text_features.clone()
        else:
            prior_mu_tensor = torch.as_tensor(
                prior_mu,
                device=device,
                dtype=dtype,
            ).clone()
            if prior_mu_tensor.shape != base_text_features.shape:
                raise ValueError(
                    "BayesPaperAdapter.prior_mu shape 不匹配："
                    f"expected {tuple(base_text_features.shape)}, "
                    f"got {tuple(prior_mu_tensor.shape)}"
                )

        min_log_sigma = math.log(1e-8)

        if self.covariance_mode == "paper_scalar":
            # (C,)
            if prior_log_sigma is None:
                prior_log_sigma_tensor = torch.full(
                    (n_classes,),
                    math.log(prior_sigma),
                    device=device,
                    dtype=dtype,
                )
            else:
                prior_log_sigma_tensor = torch.as_tensor(
                    prior_log_sigma,
                    device=device,
                    dtype=dtype,
                ).flatten().clone()
                if prior_log_sigma_tensor.shape != (n_classes,):
                    raise ValueError(
                        "paper_scalar 模式下 prior_log_sigma 必须是形状 (C,)："
                        f"expected {(n_classes,)}, got {tuple(prior_log_sigma_tensor.shape)}"
                    )

            prior_log_sigma_tensor = torch.clamp(prior_log_sigma_tensor, min=min_log_sigma)

            # 默认保持原逻辑：posterior 初值 = prior 初值
            self.variational_mu = nn.Parameter(prior_mu_tensor.clone())
            self.variational_log_sigma = nn.Parameter(prior_log_sigma_tensor.clone())

            self.register_buffer("prior_mu", prior_mu_tensor.clone())
            self.register_buffer("prior_log_sigma", prior_log_sigma_tensor.clone())

        else:
            # diag 模式：prior_log_sigma 形状优先要求 (C, D)
            # 为了兼容，也允许传 (C,) ，会自动广播成 (C, D)
            if prior_log_sigma is None:
                prior_log_sigma_tensor = torch.full(
                    (n_classes, feat_dim),
                    math.log(prior_sigma),
                    device=device,
                    dtype=dtype,
                )
            else:
                prior_log_sigma_tensor = torch.as_tensor(
                    prior_log_sigma,
                    device=device,
                    dtype=dtype,
                ).clone()

                if prior_log_sigma_tensor.shape == (n_classes,):
                    prior_log_sigma_tensor = (
                        prior_log_sigma_tensor[:, None]
                        .expand(-1, feat_dim)
                        .clone()
                    )
                elif prior_log_sigma_tensor.shape != (n_classes, feat_dim):
                    raise ValueError(
                        "diag 模式下 prior_log_sigma 必须是形状 (C, D) 或 (C,)："
                        f"expected {(n_classes, feat_dim)} or {(n_classes,)}, "
                        f"got {tuple(prior_log_sigma_tensor.shape)}"
                    )

            prior_log_sigma_tensor = torch.clamp(prior_log_sigma_tensor, min=min_log_sigma)

            self.variational_mu = nn.Parameter(prior_mu_tensor.clone())
            self.variational_log_sigma = nn.Parameter(prior_log_sigma_tensor.clone())

            self.register_buffer("prior_mu", prior_mu_tensor.clone())
            self.register_buffer("prior_log_sigma", prior_log_sigma_tensor.clone())

        self.train_mc_samples = train_mc_samples
        self.eval_mc_samples = eval_mc_samples
        self.total_epochs = total_epochs
        self.kl_scale_divisor = kl_scale_divisor
        self.kl_weight = 0.0

    def set_epoch(self, epoch: int) -> None:
        """
        尽量保持和你当前 paper-faithful 逻辑一致：
            kl_weight = (epoch / num_epochs) * 1/(1000 * C * D)
        """
        epoch0 = max(int(epoch) - 1, 0)
        epoch0 = min(epoch0, self.total_epochs)

        n_classes, feat_dim = self.variational_mu.shape
        self.kl_weight = float(epoch0 / float(self.total_epochs)) * (
            1.0 / (self.kl_scale_divisor * float(n_classes * feat_dim))
        )

    def sample_prototypes(self, n_samples: Optional[int] = None) -> torch.Tensor:
        n_samples = int(
            n_samples
            or (self.train_mc_samples if self.training else self.eval_mc_samples)
        )

        eps = torch.randn(
            (n_samples, *self.variational_mu.shape),
            device=self.variational_mu.device,
            dtype=self.variational_mu.dtype,
        )

        sigma = torch.exp(self.variational_log_sigma)

        if self.covariance_mode == "paper_scalar":
            # sigma: (C,) -> (1, C, 1)
            sigma = sigma.view(1, -1, 1)
        else:
            # sigma: (C, D) -> (1, C, D)
            sigma = sigma.unsqueeze(0)

        return self.variational_mu.unsqueeze(0) + eps * sigma

    def kl_divergence(self) -> torch.Tensor:
        """
        说明：
        - paper_scalar 分支保持你当前仓库的 paper-faithful 写法；
        - diag 分支采用同风格的按维对角 KL 写法；
        - 与 paper 分支一样，省略了只差一个常数的项，不影响梯度方向。
        """
        prior_mu = self.prior_mu.to(self.variational_mu.device)

        if self.covariance_mode == "paper_scalar":
            posterior_std = torch.exp(self.variational_log_sigma) + 1e-8   # (C,)
            prior_std = torch.exp(self.prior_log_sigma).to(self.variational_mu.device) + 1e-8
            _, feat_dim = self.variational_mu.shape

            kl_trace = feat_dim * (posterior_std.pow(2) / prior_std.pow(2)).sum()
            kl_diff_sq = (
                (self.variational_mu - prior_mu).pow(2) / prior_std.pow(2)[:, None]
            ).sum()
            kl_logdet = feat_dim * (
                prior_std.pow(2).log() - posterior_std.pow(2).log()
            ).sum()

            return kl_trace + kl_diff_sq + kl_logdet

        # diag 模式
        posterior_std = torch.exp(self.variational_log_sigma) + 1e-8        # (C, D)
        prior_std = torch.exp(self.prior_log_sigma).to(self.variational_mu.device) + 1e-8

        kl_trace = (posterior_std.pow(2) / prior_std.pow(2)).sum()
        kl_diff_sq = (
            (self.variational_mu - prior_mu).pow(2) / prior_std.pow(2)
        ).sum()
        kl_logdet = (
            prior_std.pow(2).log() - posterior_std.pow(2).log()
        ).sum()

        return kl_trace + kl_diff_sq + kl_logdet

    def mc_logits(self, image_features, logit_scale, n_samples=None):
        n_samples = int(
            n_samples or (self.train_mc_samples if self.training else self.eval_mc_samples)
        )

        image_features_norm = self._normalize_features(image_features)

        prototypes = self.sample_prototypes(n_samples=n_samples)   # [S, C, D]
        prototypes = self._normalize_features(prototypes)
        prototypes = prototypes.to(
            device=image_features_norm.device,
            dtype=image_features_norm.dtype,
        )

        scale = self._exp_scale(
            logit_scale,
            image_features_norm.dtype,
            image_features_norm.device,
        )

        # official semantics: [S, B, C]
        logits = torch.einsum("bd,scd->sbc", image_features_norm, prototypes) * scale
        return logits

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.mc_logits(
            image_features=image_features,
            logit_scale=logit_scale,
            n_samples=self.train_mc_samples if self.training else self.eval_mc_samples,
        )

    def regularization_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        kl_raw = self.kl_divergence()
        loss = kl_raw * float(self.kl_weight)
        return loss, {
            "loss_kl_raw": float(kl_raw.detach().item()),
            "kl_weight": float(self.kl_weight),
            "loss_kl": float(loss.detach().item()),
        }


class VMFPrototypeAdapter(AdapterMethod):
    """
    Training-free spherical prototype adapter。

    - posterior_eta: [C, D], already includes
        template-text prior + support updates
    - A_img_inv / B_img_inv: BayesVLM image-side Kronecker covariance factors
    - query-time uncertainty is computed from cached image activations

    注意：
    1) 这里不走 text-only run prior，text prior 已在 family 侧直接由模板 prompts 构造好；
    2) 这里优先使用 scipy.special.ive 稳定计算 log C_d(kappa)；
       如果环境里没有 scipy，则退回到原来的近似实现；
    3) forward 需要 activations，因此通过 forward_with_aux 调用。
    """

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "VMFPROTO",
        posterior_eta: Optional[torch.Tensor] = None,
        A_img_inv: Optional[torch.Tensor] = None,
        B_img_inv: Optional[torch.Tensor] = None,
        kappa_scale: float = 1.0,
        eps: float = 1e-6,
        kappa_max: float = 500.0,
    ):
        super().__init__(initialization)

        if posterior_eta is None:
            raise ValueError("VMFPrototypeAdapter requires posterior_eta")
        if A_img_inv is None or B_img_inv is None:
            raise ValueError("VMFPrototypeAdapter requires A_img_inv and B_img_inv")

        posterior_eta = torch.as_tensor(
            posterior_eta,
            device=base_text_features.device,
            dtype=base_text_features.dtype,
        )
        A_img_inv = torch.as_tensor(
            A_img_inv,
            device=base_text_features.device,
            dtype=base_text_features.dtype,
        )
        B_img_inv = torch.as_tensor(
            B_img_inv,
            device=base_text_features.device,
            dtype=base_text_features.dtype,
        )

        if posterior_eta.ndim != 2:
            raise ValueError(f"posterior_eta must be [C, D], got {tuple(posterior_eta.shape)}")
        if A_img_inv.ndim != 2 or B_img_inv.ndim != 2:
            raise ValueError("A_img_inv and B_img_inv must be matrices")

        self.register_buffer("posterior_eta", posterior_eta.clone())
        self.register_buffer("A_img_inv", A_img_inv.clone())
        self.register_buffer("B_img_inv", B_img_inv.clone())

        self.kappa_scale = float(kappa_scale)
        self.eps = float(eps)
        self.kappa_max = float(kappa_max)
        self.feat_dim = int(posterior_eta.shape[-1])

        eta_norm = posterior_eta.norm(dim=-1).clamp_min(self.eps)
        mu_post = posterior_eta / eta_norm.unsqueeze(-1)
        kappa_post = eta_norm

        self.register_buffer("kappa_post", kappa_post)
        self.register_buffer("mu_post", mu_post)
        self.register_buffer("logC_post", self._log_vmf_C(kappa_post, self.feat_dim))

        self._debug_printed = False
        self._logc_backend = "scipy_ive" if (np is not None and scipy_ive is not None) else "fallback_asymptotic"

        print(
            f"[vmf/init] feat_dim={self.feat_dim} "
            f"logC_backend={self._logc_backend} "
            f"kappa_post_min={self.kappa_post.min().item():.4f} "
            f"kappa_post_mean={self.kappa_post.mean().item():.4f} "
            f"kappa_post_max={self.kappa_post.max().item():.4f}"
        )

    def _normalize_rows(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _approx_log_vmf_C_fallback(self, kappa: torch.Tensor, dim: int) -> torch.Tensor:
        """
        中文说明：
        这是无 scipy 时的后备近似实现。
        只用于环境缺少 scipy 的情况。
        """
        kappa = kappa.clamp_min(self.eps)
        dtype = kappa.dtype
        device = kappa.device

        half_d = torch.tensor(0.5 * float(dim), device=device, dtype=dtype)
        log_c0 = torch.lgamma(half_d) - math.log(2.0) - half_d * math.log(math.pi)
        log_c0 = log_c0.expand_as(kappa)

        approx = 0.5 * float(dim - 1) * (torch.log(kappa) - math.log(2.0 * math.pi)) - kappa
        return torch.where(kappa < 1e-4, log_c0, approx)

    def _log_vmf_C(self, kappa: torch.Tensor, dim: int) -> torch.Tensor:
        """
        稳定计算 log C_d(kappa)

        C_d(kappa) = kappa^{nu} / ((2pi)^{d/2} I_nu(kappa))
        其中 nu = d/2 - 1

        数值策略：
        1) 很小 kappa：使用 kappa->0 的极限（均匀球面）
        2) 其他区间：优先使用 scipy.special.ive
           ive(nu, kappa) = exp(-|kappa|) * I_nu(kappa)
           对正 kappa，有 log I_nu(kappa) = log ive(nu, kappa) + kappa
        3) 如果 scipy 不可用，则退回到 fallback 近似
        """
        kappa = kappa.clamp_min(self.eps)

        if np is None or scipy_ive is None:
            return self._approx_log_vmf_C_fallback(kappa, dim)

        dtype = kappa.dtype
        device = kappa.device
        nu = 0.5 * float(dim) - 1.0

        half_d = torch.tensor(0.5 * float(dim), device=device, dtype=torch.float64)
        log_c0 = torch.lgamma(half_d) - math.log(2.0) - half_d * math.log(math.pi)

        kappa_np = (
            kappa.detach()
            .to(dtype=torch.float64)
            .cpu()
            .contiguous()
            .view(-1)
            .numpy()
        )

        out = np.empty_like(kappa_np, dtype=np.float64)

        small_mask = kappa_np < 1e-6
        out[small_mask] = float(log_c0.item())

        if (~small_mask).any():
            ks = kappa_np[~small_mask]
            ive_val = scipy_ive(nu, ks)
            ive_val = np.maximum(ive_val, 1e-300)
            log_I = np.log(ive_val) + ks
            out[~small_mask] = (
                nu * np.log(ks)
                - 0.5 * float(dim) * np.log(2.0 * np.pi)
                - log_I
            )

        out = torch.from_numpy(out.reshape(tuple(kappa.shape))).to(device=device, dtype=dtype)
        return out

    def _query_kappa(
        self,
        image_features: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        mu = image_features.float()
        u = self._normalize_rows(mu)

        acts = activations.float()
        alpha = torch.einsum("bi,ij,bj->b", acts, self.A_img_inv.float(), acts).clamp_min(0.0)

        tr_B = torch.trace(self.B_img_inv.float())
        Bu = torch.matmul(u, self.B_img_inv.float())
        uBu = (Bu * u).sum(dim=-1)

        mu_norm2 = (mu * mu).sum(dim=-1).clamp_min(self.eps)
        rho = alpha * (tr_B - uBu).clamp_min(0.0) / (
            (float(self.feat_dim - 1) * mu_norm2) + self.eps
        )
        kappa = self.kappa_scale / (rho + self.eps)
        return kappa.clamp(min=self.eps, max=self.kappa_max)

    def forward_with_aux(
        self,
        image_features: torch.Tensor,
        activations: torch.Tensor | None,
        residuals: torch.Tensor | None,
        logit_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del residuals, logit_scale

        if activations is None:
            raise ValueError("VMFPrototypeAdapter requires activations for query-time uncertainty")

        u = self._normalize_rows(image_features.float())                  # [B, D]
        kappa_q = self._query_kappa(image_features, activations)          # [B]

        eta = self.posterior_eta.to(device=u.device, dtype=u.dtype)       # [C, D]
        r = torch.linalg.norm(
            kappa_q[:, None, None] * u[:, None, :] + eta[None, :, :],
            dim=-1,
        )                                                                 # [B, C]

        logits = self.logC_post.to(device=r.device, dtype=r.dtype).unsqueeze(0) - self._log_vmf_C(r, self.feat_dim)

        if not self._debug_printed:
            finite_logits = torch.isfinite(logits)
            print(
                f"[vmf/query] kappa_q_min={kappa_q.min().item():.4f} "
                f"kappa_q_mean={kappa_q.mean().item():.4f} "
                f"kappa_q_max={kappa_q.max().item():.4f}"
            )
            print(
                f"[vmf/query] r_min={r.min().item():.4f} "
                f"r_mean={r.mean().item():.4f} "
                f"r_max={r.max().item():.4f}"
            )
            print(
                f"[vmf/query] logit_min={logits.min().item():.4f} "
                f"logit_mean={logits.mean().item():.4f} "
                f"logit_max={logits.max().item():.4f} "
                f"finite_ratio={finite_logits.float().mean().item():.6f}"
            )
            self._debug_printed = True




        return logits

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        del image_features, logit_scale
        raise RuntimeError("VMFPrototypeAdapter must be called through forward_with_aux(...)")

    def regularization_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = torch.zeros((), device=self.posterior_eta.device, dtype=torch.float32)
        return zero, {}




class UATangentBayesAdapterMin(BayesPaperAdapter):
    """
    Minimal falsifiable version of:
        BayesAdapter
        + tangent-anisotropic prior
        + shared feature-uncertainty gate lambda_u

    设计原则
    --------
    1) posterior 参数化保持与 BayesPaperAdapter(paper_scalar) 一致：
       q(w_c) = N(mu_c, sigma_c^2 I)
    2) prior 改为围绕 normalized prior_mu 的切空间各向异性高斯：
       p(w_c) = N(t_c, sigma_parallel^2 t_c t_c^T + sigma_perp^2 (I - t_c t_c^T))
    3) logits 仍然返回 [S, B, C]，保持你当前 trainer / loss 实现不变
    4) 当
         - sigma_parallel == sigma_perp == bayesadapter_prior_sigma
         - lambda_u == 0
       时，KL 与 forward 都退化回当前 BayesAdapter

    当前最小版本限制
    ----------------
    - 只支持 covariance_mode == "paper_scalar"
    - feature uncertainty 使用 BayesVLM image-side Hessian payload:
        alpha_n = a_n^T A_img_inv a_n
        var(logit_{s,n,c}) ~ scale^2 * lambda_u * alpha_n * w_{s,c}^T B_img_inv w_{s,c}
    """

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "UATB_MIN",
        prior_sigma: float = 0.01,
        train_mc_samples: int = 3,
        eval_mc_samples: int = 10,
        total_epochs: int = 300,
        kl_scale_divisor: float = 1000.0,
        covariance_mode: str = "paper_scalar",
        prior_mu: Optional[torch.Tensor] = None,
        prior_log_sigma: Optional[torch.Tensor] = None,
        prior_sigma_parallel: float = 0.005,
        prior_sigma_perp: float = 0.02,
        use_feature_uncertainty: bool = True,
        lambda_u_init: float = 0.0,
        lambda_u_max: float = 1.0,
        lambda_u_learnable: bool = True,
        A_img_inv: Optional[torch.Tensor] = None,
        B_img_inv: Optional[torch.Tensor] = None,
    ):
        if str(covariance_mode).lower() != "paper_scalar":
            raise ValueError("UATangentBayesAdapterMin only supports covariance_mode='paper_scalar'")

        super().__init__(
            base_text_features=base_text_features,
            initialization=initialization,
            prior_sigma=prior_sigma,
            train_mc_samples=train_mc_samples,
            eval_mc_samples=eval_mc_samples,
            total_epochs=total_epochs,
            kl_scale_divisor=kl_scale_divisor,
            covariance_mode="paper_scalar",
            prior_mu=prior_mu,
            prior_log_sigma=prior_log_sigma,
        )
        self._last_forward_stats: Dict[str, float] = {}
        if A_img_inv is None or B_img_inv is None:
            raise ValueError("UATB_MIN requires A_img_inv and B_img_inv")

        self.use_feature_uncertainty = bool(use_feature_uncertainty)
        self.prior_sigma_parallel = float(max(prior_sigma_parallel, 1e-8))
        self.prior_sigma_perp = float(max(prior_sigma_perp, 1e-8))
        self.lambda_u_max = float(max(lambda_u_max, 1e-8))

        # 方向化 prior center：围绕 normalized prior_mu 做切空间分解
        prior_dirs = self._normalize_features(self.prior_mu.detach())
        self.register_buffer("prior_dirs", prior_dirs)

        # BayesVLM image-side Hessian payload
        self.register_buffer(
            "A_img_inv",
            torch.as_tensor(A_img_inv, device=self.variational_mu.device, dtype=self.variational_mu.dtype).clone(),
        )
        self.register_buffer(
            "B_img_inv",
            torch.as_tensor(B_img_inv, device=self.variational_mu.device, dtype=self.variational_mu.dtype).clone(),
        )




        if lambda_u_init < 0.0:
            raise ValueError("lambda_u_init must be >= 0")

        self.lambda_u_eps = 1e-12
        self.lambda_u_max = float(max(lambda_u_max, self.lambda_u_eps))

        if lambda_u_init == 0.0:
            raw_init = -50.0   # softplus(-50) ~ 0
        else:
            init_val = float(min(lambda_u_init, self.lambda_u_max))
            raw_init = math.log(math.expm1(init_val))

        raw_tensor = torch.tensor(
            raw_init,
            device=self.variational_mu.device,
            dtype=self.variational_mu.dtype,
        )

        if lambda_u_learnable:
            self.raw_lambda_u = nn.Parameter(raw_tensor)
        else:
            self.register_buffer("raw_lambda_u", raw_tensor)



        raw_tensor = torch.tensor(
            raw_init,
            device=self.variational_mu.device,
            dtype=self.variational_mu.dtype,
        )
        if lambda_u_learnable:
            self.raw_lambda_u = nn.Parameter(raw_tensor)
        else:
            self.register_buffer("raw_lambda_u", raw_tensor)


    def _lambda_u(self) -> torch.Tensor:
        lam = F.softplus(self.raw_lambda_u)
        lam = torch.clamp(lam, min=0.0, max=self.lambda_u_max)
        return lam


    def _prior_geometry_stats(self) -> Dict[str, torch.Tensor]:
        delta = self.variational_mu - self.prior_mu
        t = self.prior_dirs

        delta_par = (delta * t).sum(dim=-1)                    # [C]
        delta_sq = delta.pow(2).sum(dim=-1)                    # [C]
        delta_perp_sq = (delta_sq - delta_par.pow(2)).clamp_min(0.0)

        return {
            "delta_par_sq_mean": delta_par.pow(2).mean(),
            "delta_perp_sq_mean": delta_perp_sq.mean(),
            "prior_dir_norm_mean": t.norm(dim=-1).mean(),
        }



    def kl_divergence(self) -> torch.Tensor:
        """
        与当前 BayesPaperAdapter 的 "paper-faithful" 风格保持一致：
        - 省略常数项，不影响优化方向
        - 当 sigma_parallel == sigma_perp == prior_sigma 时，
          精确退化回当前 BayesPaperAdapter.paper_scalar 的 KL 形式
        """
        posterior_std = torch.exp(self.variational_log_sigma) + 1e-8   # [C]
        delta = self.variational_mu - self.prior_mu                    # [C, D]
        t = self.prior_dirs                                            # [C, D]

        delta_par = (delta * t).sum(dim=-1)                            # [C]
        delta_sq = delta.pow(2).sum(dim=-1)                            # [C]
        delta_perp_sq = (delta_sq - delta_par.pow(2)).clamp_min(0.0)   # [C]

        _, feat_dim = self.variational_mu.shape
        sig_par2 = self.prior_sigma_parallel ** 2
        sig_perp2 = self.prior_sigma_perp ** 2
        post2 = posterior_std.pow(2)

        trace_term = post2 * (1.0 / sig_par2 + (float(feat_dim - 1) / sig_perp2))
        quad_term = delta_par.pow(2) / sig_par2 + delta_perp_sq / sig_perp2
        logdet_term = math.log(sig_par2) + float(feat_dim - 1) * math.log(sig_perp2) - feat_dim * torch.log(post2)

        return (trace_term + quad_term + logdet_term).sum()

    def forward_with_aux(
        self,
        image_features: torch.Tensor,
        activations: torch.Tensor | None,
        residuals: torch.Tensor | None,
        logit_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del residuals

        if logit_scale is None:
            raise ValueError("UATB_MIN requires logit_scale")

        n_samples = int(self.train_mc_samples if self.training else self.eval_mc_samples)

        image_features_norm = self._normalize_features(image_features)
        prototypes = self.sample_prototypes(n_samples=n_samples)   # [S, C, D]
        prototypes = self._normalize_features(prototypes)
        prototypes = prototypes.to(
            device=image_features_norm.device,
            dtype=image_features_norm.dtype,
        )

        scale = self._exp_scale(
            logit_scale,
            image_features_norm.dtype,
            image_features_norm.device,
        )

        mean_logits = torch.einsum("bd,scd->sbc", image_features_norm, prototypes) * scale

        # 允许无 activations 时退化为当前 BayesAdapter 语义，避免影响其他调用路径
        if (not self.use_feature_uncertainty) or activations is None:
            return mean_logits

        # alpha_n = a_n^T A^{-1} a_n, shape [B]
        alpha = torch.einsum(
            "bi,ij,bj->b",
            activations.float(),
            self.A_img_inv.float(),
            activations.float(),
        ).clamp_min(0.0)

        # w^T B^{-1} w, shape [S, C]
        wBw = torch.einsum(
            "scd,df,scf->sc",
            prototypes.float(),
            self.B_img_inv.float(),
            prototypes.float(),
        ).clamp_min(0.0)

        lambda_u = self._lambda_u().float()
        scale_sq = scale.float().pow(2)

        var_logits = scale_sq * lambda_u * alpha[None, :, None] * wBw[:, None, :]
        var_logits = var_logits.clamp_min(0.0)

        # probit-style variance correction
        corrected = mean_logits / torch.sqrt(
            1.0 + (math.pi / 8.0) * var_logits.to(dtype=mean_logits.dtype)
        )


        with torch.no_grad():
            correction = (mean_logits - corrected).abs().mean()
            self._last_forward_stats = {
                "alpha_mean": float(alpha.mean().item()),
                "wBw_mean": float(wBw.mean().item()),
                "var_logits_mean": float(var_logits.mean().item()),
                "var_logits_max": float(var_logits.max().item()),
                "correction_abs_mean": float(correction.item()),
            }


        return corrected

    def regularization_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats = getattr(self, "_last_forward_stats", {})
        kl_raw = self.kl_divergence()
        loss = kl_raw * float(self.kl_weight)

        geo = self._prior_geometry_stats()
        return loss, {
            "loss_kl_raw": float(kl_raw.detach().item()),
            "kl_weight": float(self.kl_weight),
            "loss_kl": float(loss.detach().item()),
            "lambda_u": float(self._lambda_u().detach().item()),
            "delta_par_sq_mean": float(geo["delta_par_sq_mean"].detach().item()),
            "delta_perp_sq_mean": float(geo["delta_perp_sq_mean"].detach().item()),
            "alpha_mean": float(stats.get("alpha_mean", 0.0)),
            "wBw_mean": float(stats.get("wBw_mean", 0.0)),
            "var_logits_mean": float(stats.get("var_logits_mean", 0.0)),
            "var_logits_max": float(stats.get("var_logits_max", 0.0)),
            "correction_abs_mean": float(stats.get("correction_abs_mean", 0.0)),
        }


ADAPTER_REGISTRY = {
    "LP": LinearProbeAdapter,
    "LINEARPROBE": LinearProbeAdapter,
    "TR": TaskResidualAdapter,
    "TASKRESIDUAL": TaskResidualAdapter,
    "CLIPA": ClipAdapter,
    "CLIPADAPTER": ClipAdapter,
    "TIPA": TipAdapter,
    "TIPADAPTER": TipAdapter,
    "CROSSMODAL": CrossModalAdapter,
    "GAUSSIAN_PER_CLASS": GaussianPerClassAdapter,
    "BAYESADAPTER": BayesPaperAdapter,
    "BAYES_ADAPTER": BayesPaperAdapter,
    "UATB_MIN": UATangentBayesAdapterMin,
    "UATANGENTBAYESADAPTERMIN": UATangentBayesAdapterMin,
    "VMFPROTO": VMFPrototypeAdapter,
    "VMF_PROTO": VMFPrototypeAdapter,
}

def build_adapter(
    adapter_name: str,
    base_text_features: torch.Tensor,
    initialization: str = "MEAN",
    **kwargs,
) -> AdapterMethod:
    key = str(adapter_name).upper()
    if key not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter_name: {adapter_name}")
    adapter_cls = ADAPTER_REGISTRY[key]
    return adapter_cls(
        base_text_features=base_text_features,
        initialization=initialization,
        **kwargs,
    )