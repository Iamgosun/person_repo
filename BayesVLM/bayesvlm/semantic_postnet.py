from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from bayesvlm.flows import SharedGaussianFlowDensity
from bayesvlm.text_priors import ClassGaussianPriors


@dataclass
class SemanticPosteriorOutput:
    alpha: torch.Tensor
    probs: torch.Tensor
    log_density: torch.Tensor
    alpha0: torch.Tensor
    aleatoric_entropy: torch.Tensor
    epistemic_inverse_evidence: torch.Tensor
    latent: torch.Tensor
    loss: Optional[torch.Tensor] = None


class SemanticPosteriorNetwork(nn.Module):
    """PostNet-style evidence classifier on top of BayesVLM text priors.

    This module merges the two papers conceptually:
      1. BayesVLM provides class-wise Gaussian text priors from prompt posteriors.
      2. A PostNet-style density model turns class densities into pseudo-counts.

    Important modeling choice:
      - The class-conditional densities remain *global* normalized densities.
      - A single image x does not modify the class density itself; it only queries the
        density at its latent location z_x. This preserves the certainty-budget logic of
        Posterior Networks.
    """

    def __init__(
        self,
        image_embed_dim: int,
        latent_dim: int,
        class_priors: ClassGaussianPriors,
        class_counts: torch.Tensor,
        projector_bias: bool = False,
        flow_layers: int = 4,
        flow_hidden_dim: int = 128,
        jitter: float = 1e-5,
        entropy_regularization: float = 1e-5,
    ):
        super().__init__()
        if class_priors.mean.ndim != 2:
            raise ValueError("class_priors.mean must have shape [C, D].")
        if class_priors.covariance.ndim != 3:
            raise ValueError("class_priors.covariance must have shape [C, D, D].")
        if class_counts.numel() != class_priors.mean.shape[0]:
            raise ValueError("class_counts and class_priors must agree on number of classes.")

        self.num_classes = class_priors.mean.shape[0]
        self.embed_dim = image_embed_dim
        self.latent_dim = latent_dim
        self.entropy_regularization = entropy_regularization
        self.jitter = jitter
        self.class_names = list(class_priors.class_names)

        self.image_projector = nn.Linear(image_embed_dim, latent_dim, bias=projector_bias)
        self.density = SharedGaussianFlowDensity(
            latent_dim=latent_dim,
            flow_layers=flow_layers,
            flow_hidden_dim=flow_hidden_dim,
            jitter=jitter,
        )

        self.register_buffer("class_counts", class_counts.float())
        self.register_buffer("text_prior_mean_full", class_priors.mean.float())
        self.register_buffer("text_prior_cov_full", class_priors.covariance.float())

    def project_text_priors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Project full-dimensional text Gaussian priors into latent space.

        If z = W h and h ~ N(mu, Sigma), then z ~ N(W mu, W Sigma W^T).
        """
        W = self.image_projector.weight  # [H, D]
        means = torch.einsum("hd,cd->ch", W, self.text_prior_mean_full)
        covs = torch.einsum("hd,cdk,lk->chl", W, self.text_prior_cov_full, W)
        eye = torch.eye(self.latent_dim, device=W.device, dtype=W.dtype)
        covs = covs + self.jitter * eye[None, :, :]
        return means, covs

    def encode_image_embeddings(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.image_projector(image_embeds)

    def compute_alpha(self, image_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode_image_embeddings(image_embeds)
        prior_means, prior_covs = self.project_text_priors()
        log_density = self.density.log_prob(z, prior_means, prior_covs)
        alpha = 1.0 + self.class_counts[None, :] * torch.exp(log_density)
        return z, log_density, alpha

    def uce_loss(self, alpha: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        alpha0 = alpha.sum(dim=-1, keepdim=True)
        digamma_alpha = torch.digamma(alpha)
        digamma_alpha0 = torch.digamma(alpha0)
        target_term = digamma_alpha0.squeeze(-1) - digamma_alpha.gather(1, labels[:, None]).squeeze(-1)
        dirichlet_entropy = torch.distributions.Dirichlet(alpha).entropy()
        loss = target_term.mean() - self.entropy_regularization * dirichlet_entropy.mean()
        return loss

    def forward(self, image_embeds: torch.Tensor, labels: Optional[torch.Tensor] = None) -> SemanticPosteriorOutput:
        latent, log_density, alpha = self.compute_alpha(image_embeds)
        alpha0 = alpha.sum(dim=-1)
        probs = alpha / alpha0[:, None]
        aleatoric_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        epistemic_inverse_evidence = 1.0 / alpha0.clamp_min(1e-12)
        loss = self.uce_loss(alpha, labels) if labels is not None else None
        return SemanticPosteriorOutput(
            alpha=alpha,
            probs=probs,
            log_density=log_density,
            alpha0=alpha0,
            aleatoric_entropy=aleatoric_entropy,
            epistemic_inverse_evidence=epistemic_inverse_evidence,
            latent=latent,
            loss=loss,
        )

    @torch.no_grad()
    def predict(self, image_embeds: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.forward(image_embeds, labels=None)
        return {
            "probs": output.probs,
            "alpha": output.alpha,
            "alpha0": output.alpha0,
            "aleatoric_entropy": output.aleatoric_entropy,
            "epistemic_inverse_evidence": output.epistemic_inverse_evidence,
            "latent": output.latent,
        }
