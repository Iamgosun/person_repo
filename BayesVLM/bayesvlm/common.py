from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoConfig


# 中文说明：
# 这里保留原仓库里常用的 projection dim 映射。
# 如果传的是本地路径，就会回退到 AutoConfig 自动推断。
PROJECTION_DIM = {
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": 512,
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": 768,
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": 1024,
}


def get_projection_dim(model_name_or_path: str, local_files_only: bool = False) -> int:
    """
    中文说明：
    支持两种情况：
    1）传入 Hugging Face repo id
    2）传入本地目录
    """
    if model_name_or_path in PROJECTION_DIM:
        return PROJECTION_DIM[model_name_or_path]

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )

    if hasattr(config, "projection_dim") and config.projection_dim is not None:
        return int(config.projection_dim)

    if hasattr(config, "text_config"):
        text_cfg = config.text_config
        if hasattr(text_cfg, "projection_dim") and text_cfg.projection_dim is not None:
            return int(text_cfg.projection_dim)

    if hasattr(config, "vision_config"):
        vision_cfg = config.vision_config
        if hasattr(vision_cfg, "projection_dim") and vision_cfg.projection_dim is not None:
            return int(vision_cfg.projection_dim)

    raise ValueError(f"Cannot infer projection_dim from: {model_name_or_path}")


def torch_load_cpu(path: str):
    """
    中文说明：
    为了兼容 CPU / GPU 环境，这里统一先加载到 CPU。
    """
    return torch.load(path, map_location="cpu")


@dataclass
class EncoderResult:
    embeds: torch.Tensor
    activations: torch.Tensor
    residuals: torch.Tensor

    def __init__(self, embeds, activations, residuals=None):
        self.embeds = embeds
        self.activations = activations
        self.residuals = residuals if residuals is not None else torch.zeros_like(embeds)

    def clone(self):
        return EncoderResult(
            embeds=self.embeds.clone(),
            activations=self.activations.clone(),
            residuals=self.residuals.clone(),
        )

    def to(self, device):
        self.embeds = self.embeds.to(device)
        self.activations = self.activations.to(device)
        self.residuals = self.residuals.to(device)
        return self

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        if isinstance(idx, (list, torch.Tensor)):
            return EncoderResult(
                embeds=self.embeds[idx],
                activations=self.activations[idx],
                residuals=self.residuals[idx],
            )
        return self.embeds[idx], self.activations[idx], self.residuals[idx]


@dataclass
class ProbabilisticLogits:
    mean: torch.Tensor
    var: torch.Tensor

    def _diag_variance(self) -> torch.Tensor:
        """
        中文说明：
        统一提取“每个类别对应的方差”。
        - 如果 var 是 [B, C]，说明本来就是逐类独立方差
        - 如果 var 是 [B, C, C]，说明是完整协方差矩阵，需要取对角线
        """
        if self.var.ndim == 2:
            return self.var.clamp_min(0.0)

        if self.var.ndim == 3:
            return self.var.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)

        raise ValueError(f"Unsupported variance shape: {self.var.shape}")

    def softmax(self, dim=-1, num_samples=400, chunk_size=10000, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        if num_samples == 0:
            # 中文说明：
            # 多分类 probit 近似。
            # 修复了旧实现里对 [B, C] 方差误用 diagonal() 的问题。
            variance = self._diag_variance()
            scaled_mean = self.mean / torch.sqrt(1 + torch.pi / 8 * variance)
            return torch.nn.functional.softmax(scaled_mean, dim=dim)

        probas = torch.zeros_like(self.mean)

        if self.var.ndim == 2:
            std = torch.sqrt(self.var.clamp_min(0.0))
            for _ in range(num_samples):
                eps = torch.randn(std.shape, device=std.device) * std
                probas += torch.nn.functional.softmax(self.mean + eps, dim=dim)

        elif self.var.ndim == 3:
            num_chunks = math.ceil(self.mean.shape[0] / chunk_size)
            mean_chunks = torch.chunk(self.mean, num_chunks, dim=0)
            var_chunks = torch.chunk(self.var, num_chunks, dim=0)

            probas = []
            for mean_chunk, var_chunk in tqdm(zip(mean_chunks, var_chunks), total=num_chunks):
                dist = torch.distributions.MultivariateNormal(
                    mean_chunk,
                    covariance_matrix=var_chunk,
                )
                probas_chunk = 0
                for _ in range(num_samples):
                    sample = dist.sample()
                    probas_chunk += torch.nn.functional.softmax(sample, dim=dim)
                probas.append(probas_chunk)

            probas = torch.concat(probas, dim=0)

        else:
            raise ValueError(f"Invalid variance tensor shape: {self.var.shape}")

        return probas / num_samples

    def sample_probas(self, num_samples: int, seed=None):
        """
        中文说明：
        从 logits 分布采样，然后转为 softmax 概率。
        返回形状：
            [N, num_samples, num_classes]
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.var.ndim == 2:
            std = torch.sqrt(self.var.clamp_min(0.0))
            samples = (
                torch.randn((num_samples,) + self.mean.shape, device=self.mean.device) * std
                + self.mean
            )
            samples = samples.permute(1, 0, 2)
            return torch.nn.functional.softmax(samples, dim=2)

        elif self.var.ndim == 3:
            dist = torch.distributions.MultivariateNormal(
                self.mean,
                covariance_matrix=self.var,
            )

            samples = []
            for _ in range(num_samples):
                sample = dist.sample((1,))
                samples.append(sample)
            samples = torch.cat(samples, dim=0)

            samples = samples.permute(1, 0, 2)
            return torch.nn.functional.softmax(samples, dim=2)

        else:
            raise ValueError(f"Invalid variance tensor shape: {self.var.shape}")

    def expected_aleatoric_entropy(self, num_samples=400, dim=-1):
        entropy = 0

        if self.var.ndim == 2:
            std = torch.sqrt(self.var.clamp_min(0.0))
            for _ in range(num_samples):
                eps = torch.randn(self.var.shape, device=self.var.device) * std
                probas = torch.nn.functional.softmax(self.mean + eps, dim=dim)
                entropy += -(probas * probas.log()).sum(dim=dim)

        elif self.var.ndim == 3:
            dist = torch.distributions.MultivariateNormal(
                self.mean,
                covariance_matrix=self.var,
            )
            for _ in range(num_samples):
                sample = dist.sample()
                probas = torch.nn.functional.softmax(sample, dim=dim)
                entropy += -(probas * probas.log()).sum(dim=dim)

        else:
            raise ValueError(f"Invalid variance tensor shape: {self.var.shape}")

        return entropy / num_samples

    def __getitem__(self, idx):
        return ProbabilisticLogits(
            mean=self.mean[idx],
            var=self.var[idx],
        )

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def detach(self):
        return ProbabilisticLogits(
            mean=self.mean.detach(),
            var=self.var.detach(),
        )

    def cross_entropy(self, target, num_samples=400, reduction="sum"):
        if num_samples == 0:
            # 中文说明：
            # 训练时默认使用解析 probit 近似，减少 MC 采样波动。
            variance = self._diag_variance()
            scaled_mean = self.mean / torch.sqrt(1 + torch.pi / 8 * variance)
            return torch.nn.functional.cross_entropy(
                scaled_mean,
                target,
                reduction=reduction,
            )

        loss = 0

        if self.var.ndim == 2:
            std = torch.sqrt(self.var.clamp_min(0.0))
            for _ in range(num_samples):
                eps = torch.randn(self.var.shape, device=self.var.device) * std
                logits = self.mean + eps
                loss += torch.nn.functional.cross_entropy(
                    logits,
                    target,
                    reduction=reduction,
                )

        elif self.var.ndim == 3:
            dist = torch.distributions.MultivariateNormal(
                self.mean,
                covariance_matrix=self.var,
            )
            for _ in range(num_samples):
                sample = dist.sample()
                loss += torch.nn.functional.cross_entropy(
                    sample,
                    target,
                    reduction=reduction,
                )

        else:
            raise ValueError(f"Invalid variance tensor shape: {self.var.shape}")

        return loss / num_samples

    def clone(self):
        return ProbabilisticLogits(
            mean=self.mean.clone(),
            var=self.var.clone(),
        )