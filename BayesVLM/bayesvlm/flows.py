from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    """Simple RealNVP-style affine coupling layer for low-dimensional latents."""

    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor, scale_clamp: float = 2.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.scale_clamp = scale_clamp
        self.register_buffer("mask", mask.view(1, dim))
        self.scale_net = MLP(dim, hidden_dim, dim)
        self.shift_net = MLP(dim, hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        log_s = torch.tanh(self.scale_net(x_masked)) * self.scale_clamp
        t = self.shift_net(x_masked)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(log_s) + t)
        log_det = ((1.0 - self.mask) * log_s).sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        log_s = torch.tanh(self.scale_net(y_masked)) * self.scale_clamp
        t = self.shift_net(y_masked)
        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-log_s))
        log_det = -((1.0 - self.mask) * log_s).sum(dim=-1)
        return x, log_det


class RealNVP(nn.Module):
    """Shared invertible flow used to reshape the latent density for all classes."""

    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 128):
        super().__init__()
        if dim < 2:
            raise ValueError("RealNVP requires latent dimension >= 2.")
        layers: List[AffineCoupling] = []
        for layer_idx in range(num_layers):
            mask = torch.zeros(dim)
            mask[layer_idx % 2 :: 2] = 1.0
            layers.append(AffineCoupling(dim=dim, hidden_dim=hidden_dim, mask=mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        h = x
        for layer in self.layers:
            h, delta = layer(h)
            log_det = log_det + delta
        return h, log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        h = z
        for layer in reversed(self.layers):
            h, delta = layer.inverse(h)
            log_det = log_det + delta
        return h, log_det


class SharedGaussianFlowDensity(nn.Module):
    """Class-conditional density with shared flow and class-specific Gaussian bases.

    For a latent vector z, the class-conditional density is

        p(z | c) = q0_c(f^{-1}(z)) * |det J_{f^{-1}}(z)|

    where q0_c is a class-specific Gaussian base and f is a shared invertible flow.
    The shared flow lets class differences come primarily from the text-derived base
    distributions instead of duplicating one flow per class.
    """

    def __init__(self, latent_dim: int, flow_layers: int = 4, flow_hidden_dim: int = 128, jitter: float = 1e-5):
        super().__init__()
        self.latent_dim = latent_dim
        self.flow = RealNVP(dim=latent_dim, num_layers=flow_layers, hidden_dim=flow_hidden_dim)
        self.jitter = jitter

    def log_prob(
        self,
        z: torch.Tensor,
        base_means: torch.Tensor,
        base_covariances: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate class-wise log probabilities.

        Args:
            z: [B, H] latent points.
            base_means: [C, H] class-specific Gaussian means.
            base_covariances: [C, H, H] class-specific Gaussian covariances.

        Returns:
            log probabilities with shape [B, C].
        """
        z0, inverse_log_det = self.flow.inverse(z)
        batch_size = z.shape[0]
        num_classes = base_means.shape[0]
        log_probs = []
        eye = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        for class_idx in range(num_classes):
            cov = base_covariances[class_idx] + self.jitter * eye
            dist = torch.distributions.MultivariateNormal(base_means[class_idx], covariance_matrix=cov)
            log_probs.append(dist.log_prob(z0) + inverse_log_det)
        return torch.stack(log_probs, dim=-1)
