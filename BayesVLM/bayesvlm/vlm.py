from __future__ import annotations

from typing import Optional

import torch
from transformers import (
    CLIPModel,
    SiglipModel,
)

from bayesvlm.common import EncoderResult, ProbabilisticLogits
from bayesvlm.hessians import KroneckerFactorizedCovariance


class CLIP(torch.nn.Module):
    """
    中文说明：
    1. 这个文件现在只保留 VLM 头部逻辑，不再混入 text/image encoder。
    2. 这样 text encoder / image encoder 可以各自独立扩展。
    3. 这里仍然保留原来的：
       - 确定性 logits 计算
       - Smith 风格的不确定性传播
    """

    source_projection_has_bias = False
    target_projection_has_bias = False

    def __init__(
        self,
        logit_scale: float,
        logit_bias: float = 0,
        source_covariance: KroneckerFactorizedCovariance = None,
        target_covariance: KroneckerFactorizedCovariance = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([], device=device) * logit_scale)
        self.logit_bias = torch.nn.Parameter(torch.ones([], device=device) * logit_bias)
        self.source_covariance = source_covariance
        self.target_covariance = target_covariance

    @property
    def device(self):
        return self.logit_scale.data.device

    def set_covariances(
        self,
        source_covariance: KroneckerFactorizedCovariance = None,
        target_covariance: KroneckerFactorizedCovariance = None,
    ):
        self.source_covariance = (
            KroneckerFactorizedCovariance(
                A_inv=source_covariance.A_inv.clone().to(self.device),
                B_inv=source_covariance.B_inv.clone().to(self.device),
            )
            if source_covariance is not None
            else None
        )

        self.target_covariance = (
            KroneckerFactorizedCovariance(
                A_inv=target_covariance.A_inv.clone().to(self.device),
                B_inv=target_covariance.B_inv.clone().to(self.device),
            )
            if target_covariance is not None
            else None
        )

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        clip = CLIPModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model = cls(
            logit_scale=clip.logit_scale.item(),
        )
        model = model.to(device) if device is not None else model
        return model

    def _compute_logits(
        self,
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ):
        # 中文说明：
        # 先做 L2 normalize，再做余弦相似度。
        source_embeds = source_embeds / source_embeds.norm(p=2, dim=-1, keepdim=True)
        target_embeds = target_embeds / target_embeds.norm(p=2, dim=-1, keepdim=True)

        similarity = torch.matmul(source_embeds, target_embeds.t())
        similarity = similarity * self.logit_scale.exp() + self.logit_bias
        return similarity

    def _compute_probabilistic_logits_smith(
        self,
        source_results: EncoderResult,
        target_results: EncoderResult,
        compute_covariance: bool = False,
    ):
        """
        中文说明：
        这里保留原仓库中的 Smith 风格近似：
        - 输入是两个带 activation 的 EncoderResult
        - 用 Kronecker 分解协方差传播到最终 similarity 的 mean / var
        """
        if compute_covariance:
            raise NotImplementedError("Only the variances are supported for now.")

        if self.source_covariance is None or self.target_covariance is None:
            raise ValueError("source_covariance 和 target_covariance 不能为空。")

        source_covariance = self.source_covariance
        target_covariance = self.target_covariance

        source_activations = source_results.activations
        target_activations = target_results.activations

        if self.source_projection_has_bias:
            source_activations = torch.cat(
                [source_activations, torch.ones_like(source_activations[:, :1])],
                dim=-1,
            )

        if self.target_projection_has_bias:
            target_activations = torch.cat(
                [target_activations, torch.ones_like(target_activations[:, :1])],
                dim=-1,
            )

        source_embeds = source_results.embeds
        target_embeds = target_results.embeds

        source_B_factor = source_covariance.B_inv.diagonal()
        target_B_factor = target_covariance.B_inv.diagonal()

        source_diag_cov = (
            torch.einsum(
                "ij,jk,ik->i",
                source_activations,
                source_covariance.A_inv,
                source_activations,
            )[:, None]
            * source_B_factor
        )

        target_diag_cov = (
            torch.einsum(
                "ij,jk,ik->i",
                target_activations,
                target_covariance.A_inv,
                target_activations,
            )[:, None]
            * target_B_factor
        )

        norm_source = source_embeds**2 + source_diag_cov
        expect_norm_source = norm_source.sum(dim=-1, keepdim=True)

        norm_target = target_embeds**2 + target_diag_cov
        expect_norm_target = norm_target.sum(dim=-1, keepdim=True)

        expected_similarity = torch.matmul(
            source_embeds / torch.sqrt(expect_norm_source),
            (target_embeds / torch.sqrt(expect_norm_target)).t(),
        )

        term1 = torch.matmul(norm_source, target_diag_cov.t())
        term2 = torch.matmul(source_diag_cov, (target_embeds**2).t())
        variance_similarity = (term1 + term2) / (expect_norm_source * expect_norm_target.t())

        scale = self.logit_scale.exp()

        return ProbabilisticLogits(
            mean=expected_similarity * scale,
            var=variance_similarity * (scale**2),
        )

    def forward(
        self,
        source_embeds: torch.Tensor | EncoderResult,
        target_embeds: torch.Tensor | EncoderResult,
        map_estimate: bool = False,
    ):
        """
        中文说明：
        - 如果输入是 Tensor，就走普通确定性 logits。
        - 如果输入是 EncoderResult，就可以走不确定性传播。
        """
        if isinstance(source_embeds, EncoderResult) and isinstance(target_embeds, EncoderResult):
            if map_estimate:
                logits_map = self._compute_logits(source_embeds.embeds, target_embeds.embeds)
                covar_map = torch.zeros_like(logits_map)
                return ProbabilisticLogits(mean=logits_map, var=covar_map)

            return self._compute_probabilistic_logits_smith(source_embeds, target_embeds)

        return self._compute_logits(source_embeds, target_embeds)


class SIGLIP(CLIP):
    source_projection_has_bias = True
    target_projection_has_bias = True

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        siglip = SiglipModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model = cls(
            logit_scale=siglip.logit_scale.item(),
            logit_bias=siglip.logit_bias.item(),
        )
        model = model.to(device) if device is not None else model
        return model