from __future__ import annotations

from typing import Literal, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


TextStateKind = Literal["vector", "distribution"]


class AdapterMethod(nn.Module):
    """Base class for deterministic VLM adapters.

    Notes
    -----
    - The current few-shot comparison setup uses deterministic CLIP features.
    - Therefore every adapter's public ``forward`` returns deterministic logits.
    - Some adapters (e.g. GaussianPerClass) still expose extra methods such as
      ``sample_prototypes`` / ``kl_divergence`` for future experiments, but the
      default inference path remains deterministic.
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

    @staticmethod
    def _normalize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _exp_scale(logit_scale: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
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

        print(f">> Unrecognized initialization '{self.initialization}', fallback to MEAN in LinearProbeAdapter")
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
    """Same parameterization as LP.

    In the reference trainer, CrossModal differs mainly in the trainer-side data
    construction (mixing image features with resampled text prototypes), not in
    the classifier head itself. So the adapter stays identical to LP.
    """

    input_kind: TextStateKind = "vector"

    def __init__(self, base_text_features: torch.Tensor, initialization: str = "CrossModal"):
        super().__init__(base_text_features=base_text_features, initialization=initialization)


class TaskResidualAdapter(AdapterMethod):
    """TaskRes-style residual adapter.

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
        base = self.base_text_features.to(device=image_features.device, dtype=image_features.dtype)
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
    def init_tipadapter(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        features_train = torch.as_tensor(features_train, device=self.base_text_features.device, dtype=self.base_text_features.dtype)
        labels_train = torch.as_tensor(labels_train, device=self.base_text_features.device, dtype=torch.long)
        one_hot = F.one_hot(labels_train, num_classes=self.base_text_features.shape[0]).to(dtype=self.base_text_features.dtype)
        self.cache_keys = nn.Parameter(features_train.clone(), requires_grad=True)
        self.cache_values = nn.Parameter(one_hot.clone(), requires_grad=False)

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_features_norm = self._normalize_features(image_features)
        base = self.base_text_features.to(device=image_features.device, dtype=image_features.dtype)
        prototypes = self._normalize_features(base)
        scale = self._exp_scale(logit_scale, image_features.dtype, image_features.device)
        logits = image_features_norm @ prototypes.t() * scale

        if self.cache_keys is not None and self.cache_values is not None:
            cache_keys = self._normalize_features(
                self.cache_keys.to(device=image_features.device, dtype=image_features.dtype)
            )
            cache_values = self.cache_values.to(device=image_features.device, dtype=image_features.dtype)
            affinity = image_features @ cache_keys.t()
            cache_logits = torch.exp((-1.0) * (self.beta - self.beta * affinity)) @ cache_values
            logits = logits + self.alpha * cache_logits.to(device=logits.device, dtype=logits.dtype)

        return logits


class GaussianPerClassAdapter(AdapterMethod):
    """Per-class Gaussian adapter.

    This keeps the variational parameters and KL term from the reference design,
    but its default forward is deterministic and uses the posterior mean
    ``variational_mu`` so it stays compatible with the deterministic CLIP
    comparison protocol.
    """

    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "GAUSSIAN_PER_CLASS",
        prior_sigma: float = 0.01,
    ):
        super().__init__(initialization)
        n_classes, feat_dim = base_text_features.shape
        dtype = base_text_features.dtype
        device = base_text_features.device

        self.variational_mu = nn.Parameter(base_text_features.clone())
        self.variational_log_sigma = nn.Parameter(
            torch.full((n_classes, feat_dim), math.log(float(prior_sigma)), device=device, dtype=dtype)
        )
        self.register_buffer("prior_mu", base_text_features.clone())
        self.register_buffer(
            "prior_sigma",
            torch.full((n_classes, feat_dim), float(prior_sigma), device=device, dtype=dtype),
        )

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        eps = torch.randn(
            (int(n_samples), *self.variational_mu.shape),
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
            (posterior_sigma.pow(2) + (self.variational_mu - prior_mu).pow(2)) / (2 * prior_sigma.pow(2))
            - 0.5
            - torch.log(posterior_sigma)
            + torch.log(prior_sigma)
        )
        return torch.clamp(kl_per_dim.sum(), max=1e4)

    def mc_logits(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
        n_samples: int = 3,
    ) -> torch.Tensor:
        image_features_norm = self._normalize_features(image_features)
        prototypes = self.sample_prototypes(n_samples=n_samples)
        prototypes = self._normalize_features(prototypes)
        prototypes = prototypes.to(device=image_features_norm.device, dtype=image_features_norm.dtype)
        scale = self._exp_scale(logit_scale, image_features_norm.dtype, image_features_norm.device)
        logits = torch.einsum("bd,scd->bsc", image_features_norm, prototypes) * scale
        return logits.mean(dim=1)

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_features_norm = self._normalize_features(image_features)
        prototypes = self._normalize_features(self.variational_mu)
        prototypes = prototypes.to(device=image_features_norm.device, dtype=image_features_norm.dtype)
        scale = self._exp_scale(logit_scale, image_features_norm.dtype, image_features_norm.device)
        return image_features_norm @ prototypes.t() * scale


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
    return adapter_cls(base_text_features=base_text_features, initialization=initialization, **kwargs)
