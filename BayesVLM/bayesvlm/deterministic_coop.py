from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from bayesvlm.coop_prompt import CoOpPromptLearner


class DeterministicCoOpModel(nn.Module):
    """
    标准 deterministic CoOp:
    - 图像侧: 直接取 image embedding
    - 文本侧: 用 CoOp prompt learner 生成每个类的文本 embedding
    - 分类头: 用原始 VLM 的 deterministic cosine logits
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_learner: CoOpPromptLearner,
        vlm: nn.Module,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_learner = prompt_learner
        self.vlm = vlm

    def _model_device(self) -> torch.device:
        return self.vlm.logit_scale.device

    def train(self, mode: bool = True):
        """
        关键点：
        CoOp 训练时只让 prompt learner 进入 train mode；
        冻结的 image/text backbone 和 vlm 头保持 eval mode。
        """
        super().train(mode)
        self.prompt_learner.train(mode)

        self.image_encoder.eval()
        self.prompt_learner.text_encoder.eval()
        self.vlm.eval()
        return self

    def encode_image_batch(
        self,
        batch=None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_embeds is None and batch is not None:
            if "image_embeds" in batch:
                image_embeds = batch["image_embeds"]
            elif "embeds" in batch:
                image_embeds = batch["embeds"]

        if image_embeds is not None:
            g = image_embeds.to(self._model_device())
        else:
            g = self.image_encoder(batch, return_activations=False)

        return g.float()

    def forward(
        self,
        batch=None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None and image_embeds is None:
            raise ValueError("batch 和 image_embeds 不能同时为空。")

        g = self.encode_image_batch(batch=batch, image_embeds=image_embeds)
        text_outputs = self.prompt_learner()
        mu = text_outputs.embeds.float()

        # 这里直接走原始 deterministic logits
        logits = self.vlm(g, mu)
        return logits

    def forward_from_features(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.forward(image_embeds=image_embeds)