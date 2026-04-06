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
    1. 图像侧保持确定性，只取图像编码器输出的 embedding。
    2. 文本侧使用 CoOp prompt 生成的激活，通过文本投影层的后验协方差做不确定性传播。
    3. 输出每个样本、每个类别的 probabilistic logits(mean / var)。
    4. 默认先支持 diag 版本；如果 use_full_cov=True，则启用 full-cov 扩展版。
    5. 新增支持直接消费缓存后的 image_embeds，而不再重复调用 image_encoder。
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
        self.use_full_cov = use_full_cov
        self.normalize_image_embeds = normalize_image_embeds

    @staticmethod
    def _append_bias_if_needed(activations: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        """
        中文说明：
        如果文本投影层带 bias，则按 BayesVLM 的写法，需要在 activation 后面补一个 1。
        """
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
        2) batch 中已带缓存好的 image_embeds / embeds
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

    def forward(
        self,
        batch=None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> ProbabilisticLogits:
        if batch is None and image_embeds is None:
            raise ValueError("batch 和 image_embeds 不能同时为空。")

        g = self.encode_image_batch(batch=batch, image_embeds=image_embeds)

        # prompt learner 返回：
        #   embeds      -> 文本类原型均值 mu_c
        #   activations -> 投影前文本激活 f_c
        text_outputs = self.prompt_learner()
        mu = text_outputs.embeds.float()               # [C, D]
        text_acts = text_outputs.activations.float()   # [C, D_txt]

        text_acts = self._append_bias_if_needed(
            activations=text_acts,
            projection=self.prompt_learner.text_encoder.text_projection,
        )

        A_inv = self.text_covariance.A_inv.to(g.device).float()
        B_inv = self.text_covariance.B_inv.to(g.device).float()

        # alpha_c = f_c^T A_inv f_c
        alpha = torch.einsum("ci,ij,cj->c", text_acts, A_inv, text_acts).clamp_min(0.0)  # [C]

        # trace(Sigma_c) = alpha_c * trace(B_inv)
        trace_B = torch.trace(B_inv)
        trace_sigma = alpha * trace_B  # [C]

        # 图像范数项
        g_norm2 = (g ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B, 1]
        g_norm = torch.sqrt(g_norm2)

        # 文本范数项
        mu_norm2 = (mu ** 2).sum(dim=-1)  # [C]
        denom_text = torch.sqrt(mu_norm2 + trace_sigma + 1e-6)  # [C]

        # 期望余弦
        mean_cos = (g @ mu.t()) / (g_norm * denom_text.unsqueeze(0))

        # 方差项
        if self.use_full_cov:
            # full-cov 扩展版：g^T B_inv g
            g_quad = torch.einsum("bi,ij,bj->b", g, B_inv, g).unsqueeze(-1)  # [B, 1]
        else:
            # diag 版：g^T diag(B_inv) g
            diag_B = torch.diagonal(B_inv)
            g_quad = ((g ** 2) * diag_B.unsqueeze(0)).sum(dim=-1, keepdim=True)  # [B, 1]

        denom_var = g_norm2 * (mu_norm2 + trace_sigma).unsqueeze(0) + 1e-6
        var_cos = (g_quad * alpha.unsqueeze(0)) / denom_var
        var_cos = var_cos.clamp_min(0.0)

        scale = self.logit_scale.exp().float()
        logits_mean = mean_cos * scale
        logits_var = var_cos * (scale ** 2)

        if self.logit_bias is not None:
            logits_mean = logits_mean + self.logit_bias.float()

        return ProbabilisticLogits(
            mean=logits_mean,
            var=logits_var,
        )

    def forward_from_features(self, image_embeds: torch.Tensor) -> ProbabilisticLogits:
        return self.forward(image_embeds=image_embeds)