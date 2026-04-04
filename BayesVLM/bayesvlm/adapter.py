import torch
import torch.nn as nn
from typing import Literal

TextStateKind = Literal["vector", "distribution"]

class AdapterMethod(nn.Module):
    input_kind: TextStateKind = "vector"

    def __init__(self, initialization: str = "MEAN"):
        super().__init__()
        self.initialization = initialization

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _normalize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class LinearProbeAdapter(AdapterMethod):
    input_kind: TextStateKind = "vector"

    def __init__(
        self,
        base_text_features: torch.Tensor,
        initialization: str = "MEAN",
    ):
        super().__init__(initialization)
        self.prototypes = nn.Parameter(self._init_prototypes(base_text_features))

    def _init_prototypes(self, base_text_features: torch.Tensor) -> torch.Tensor:
        if self.initialization == "RANDOM":
            print(">> Using RANDOM initialization in LinearProbeAdapter")
            init_weight = torch.empty_like(base_text_features)
            nn.init.kaiming_normal_(init_weight)
            return init_weight

        print(">> Using MEAN initialization in LinearProbeAdapter")
        return base_text_features.clone()

    def forward(
        self,
        image_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self._normalize_features(image_features)
        prototypes = self._normalize_features(self.prototypes)
        scale = logit_scale.exp()
        return image_features @ prototypes.t() * scale