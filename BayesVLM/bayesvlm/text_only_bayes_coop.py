from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from bayesvlm.common import ProbabilisticLogits
from bayesvlm.coop_prompt import CoOpPromptLearner
from bayesvlm.hessians import KroneckerFactorizedCovariance


class TextOnlyBayesCoOpModel(nn.Module):
    """
    中文说明：
    1. 图像侧保持确定性，只取 image embedding。
    2. 文本侧由 CoOp prompt 产生确定性的 pooled activation，再通过文本投影层的后验协方差
       推出类别原型的不确定性。
    3. 训练时同时支持：
       - MAP / deterministic CoOp logits
       - Bayes predictive logits
    4. 评估与导出默认走 Bayes predictive。
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_learner: CoOpPromptLearner,
        text_covariance: KroneckerFactorizedCovariance,
        logit_scale: torch.nn.Parameter,
        logit_bias: Optional[torch.nn.Parameter] = None,
        use_full_cov: bool = False,
        normalize_image_embeds: bool = False,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_learner = prompt_learner
        self.text_covariance = text_covariance
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias
        self.use_full_cov = bool(use_full_cov)
        self.normalize_image_embeds = bool(normalize_image_embeds)

    def train(self, mode: bool = True):
        """
        中文说明：
        只让 prompt learner 进入 train mode；
        冻结的 image/text backbone 始终保持 eval mode，避免 dropout / BN 干扰。
        """
        super().train(mode)
        self.prompt_learner.train(mode)
        self.image_encoder.eval()
        self.prompt_learner.text_encoder.eval()
        return self

    @staticmethod
    def _append_bias_if_needed(activations: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        has_bias = getattr(projection, "bias", None) is not None
        if not has_bias:
            return activations

        ones = torch.ones(
            activations.shape[0],
            1,
            device=activations.device,
            dtype=activations.dtype,
        )
        return torch.cat([activations, ones], dim=-1)

    def _model_device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image_batch(self, batch=None, image_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        优先级：
        1) 显式传入 image_embeds
        2) batch 中已有缓存好的 image_embeds / embeds
        3) 否则回退到原始 image_encoder(batch)
        """
        if image_embeds is None and batch is not None:
            if "image_embeds" in batch:
                image_embeds = batch["image_embeds"]
            elif "embeds" in batch:
                image_embeds = batch["embeds"]

        if image_embeds is not None:
            g = image_embeds.to(self._model_device())
        else:
            g = self.image_encoder(batch, return_activations=False)

        g = g.float()
        if self.normalize_image_embeds:
            g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return g

    def compute_text_statistics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
            mu:          [C, D]     类别原型的 MAP 均值
            text_acts:   [C, D_txt] 投影前激活（必要时已补 bias）
            alpha:       [C]        f_c^T A_inv f_c
            trace_sigma: [C]        tr(Sigma_c) = alpha_c * tr(B_inv)
        """
        text_outputs = self.prompt_learner()
        mu = text_outputs.embeds.float()
        text_acts = text_outputs.activations.float()

        text_acts = self._append_bias_if_needed(
            activations=text_acts,
            projection=self.prompt_learner.text_encoder.text_projection,
        )

        A_inv = self.text_covariance.A_inv.to(mu.device).float()
        B_inv = self.text_covariance.B_inv.to(mu.device).float()

        alpha = torch.einsum("ci,ij,cj->c", text_acts, A_inv, text_acts).clamp_min(0.0)
        trace_sigma = alpha * torch.trace(B_inv)
        return mu, text_acts, alpha, trace_sigma

    def forward_map_logits(
        self,
        batch=None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        中文说明：
        标准 deterministic CoOp / MAP logits。
        这是“只用 prompt 做任务适配”的稳定训练目标。
        """
        if batch is None and image_embeds is None:
            raise ValueError("batch 和 image_embeds 不能同时为空。")

        g = self.encode_image_batch(batch=batch, image_embeds=image_embeds)
        mu, _, _, _ = self.compute_text_statistics()

        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        scale = self.logit_scale.exp().float()
        logits = (g @ mu.t()) * scale

        if self.logit_bias is not None:
            logits = logits + self.logit_bias.float()

        return logits

    def forward_bayes_logits(
        self,
        batch=None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> ProbabilisticLogits:
        """
        中文说明：
        Bayes predictive logits(mean / var)。
        这里不是简单给 deterministic logits 额外挂一个常数方差，
        因为：
        - mean 依赖 mu 和 trace(Sigma_c)
        - var 依赖 alpha_c 与 g^T B_inv g
        所以 prompt 改变时，均值项和方差项都会随之变化。
        """
        if batch is None and image_embeds is None:
            raise ValueError("batch 和 image_embeds 不能同时为空。")

        g = self.encode_image_batch(batch=batch, image_embeds=image_embeds)
        mu, _, alpha, trace_sigma = self.compute_text_statistics()

        B_inv = self.text_covariance.B_inv.to(g.device).float()

        g_norm2 = (g ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        g_norm = torch.sqrt(g_norm2)

        mu_norm2 = (mu ** 2).sum(dim=-1).clamp_min(1e-6)
        denom_text = torch.sqrt(mu_norm2 + trace_sigma + 1e-6)

        mean_cos = (g @ mu.t()) / (g_norm * denom_text.unsqueeze(0))

        if self.use_full_cov:
            g_quad = torch.einsum("bi,ij,bj->b", g, B_inv, g).unsqueeze(-1)
        else:
            diag_B = torch.diagonal(B_inv)
            g_quad = ((g ** 2) * diag_B.unsqueeze(0)).sum(dim=-1, keepdim=True)

        denom_var = g_norm2 * (mu_norm2 + trace_sigma).unsqueeze(0) + 1e-6
        var_cos = (g_quad * alpha.unsqueeze(0)) / denom_var
        var_cos = var_cos.clamp_min(0.0)

        scale = self.logit_scale.exp().float()
        logits_mean = mean_cos * scale
        logits_var = var_cos * (scale ** 2)

        if self.logit_bias is not None:
            logits_mean = logits_mean + self.logit_bias.float()

        return ProbabilisticLogits(mean=logits_mean, var=logits_var)

    def forward(
        self,
        batch=None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> ProbabilisticLogits:
        return self.forward_bayes_logits(batch=batch, image_embeds=image_embeds)

    def forward_from_features(self, image_embeds: torch.Tensor) -> ProbabilisticLogits:
        return self.forward(image_embeds=image_embeds)
