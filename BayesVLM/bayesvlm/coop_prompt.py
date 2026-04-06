from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from bayesvlm.common import EncoderResult
from bayesvlm.text_encoder import CLIPTextEncoder, SiglipTextEncoder


class CoOpPromptLearner(nn.Module):
    """
    中文说明：
    1. 这是 unified context 版本的 CoOp prompt learner。
    2. 只学习一组共享上下文向量 self.ctx。
    3. 类别差异只来自类别名 token。
    4. 这里依赖解耦后的 text_encoder.py 提供的 embedding 级前向接口。
    """


    def __init__(
        self,
        class_names: Sequence[str],
        text_encoder: CLIPTextEncoder | SiglipTextEncoder,
        n_ctx: int = 16,
        ctx_init: str = "a photo of",
        class_token_position: str = "end",
        fixed_suffix: str = "",
    ):



        super().__init__()
        self.class_names = [str(name).replace("_", " ") for name in class_names]
        self.text_encoder = text_encoder
        self.n_ctx = int(n_ctx)
        self.class_token_position = class_token_position
        self.fixed_suffix = fixed_suffix
        self.tokenizer = text_encoder.tokenizer
        self.token_embedding = text_encoder.get_token_embedding_layer()


        self.max_length = int(getattr(self.tokenizer, "model_max_length", 77))
        self.bos_token_id = int(self._get_special_token_id("bos_token_id", fallback="cls_token_id", default=0))
        self.eos_token_id = int(self._get_special_token_id("eos_token_id", fallback="sep_token_id", default=2))
        self.pad_token_id = int(self._get_special_token_id("pad_token_id", fallback="eos_token_id", default=self.eos_token_id))

        # 中文说明：
        # 用自然语言 prompt 初始化共享上下文；如果长度不够，则随机补齐。
        init_ctx = self._init_ctx_vectors(
            ctx_init=ctx_init,
            n_ctx=self.n_ctx,
        )
        self.ctx = nn.Parameter(init_ctx)

        # 中文说明：
        # 预先缓存每个类别名的 token ids（不带 special tokens），后续每个 epoch 重复复用。
        self.name_token_ids: List[torch.Tensor] = []
        for name in self.class_names:
            tokenized = self.tokenizer(
                name,
                add_special_tokens=False,
                return_tensors="pt",
            )
            ids = tokenized["input_ids"][0].detach().cpu()
            self.name_token_ids.append(ids)


        # 固定后缀 token（不带 special tokens）
        if self.fixed_suffix is not None and len(self.fixed_suffix.strip()) > 0:
            suffix_tokenized = self.tokenizer(
                self.fixed_suffix,
                add_special_tokens=False,
                return_tensors="pt",
            )
            self.suffix_token_ids = suffix_tokenized["input_ids"][0].detach().cpu()
        else:
            self.suffix_token_ids = torch.empty(0, dtype=torch.long)




    def _get_special_token_id(self, primary: str, fallback: str | None = None, default: int = 0) -> int:
        value = getattr(self.tokenizer, primary, None)
        if value is None and fallback is not None:
            value = getattr(self.tokenizer, fallback, None)
        if value is None:
            value = default
        return int(value)

    def _device(self) -> torch.device:
        return self.ctx.device

    def _dtype(self) -> torch.dtype:
        return self.ctx.dtype

    def _init_ctx_vectors(
        self,
        ctx_init: str,
        n_ctx: int,
    ) -> torch.Tensor:
        embed_weight = self.token_embedding.weight
        device = embed_weight.device
        dtype = embed_weight.dtype
        embed_dim = embed_weight.shape[1]

        if ctx_init is None or len(ctx_init.strip()) == 0:
            return torch.empty(n_ctx, embed_dim, device=device, dtype=dtype).normal_(std=0.02)

        tokenized = self.tokenizer(
            ctx_init,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = tokenized["input_ids"][0].to(device)
        attention_mask = tokenized.get("attention_mask", None)

        if attention_mask is None:
            valid_len = input_ids.numel()
        else:
            valid_len = int(attention_mask[0].sum().item())

        # 中文说明：
        # 尽量去掉首尾 special tokens，仅保留正文 token 作为上下文初始化。
        content_ids = input_ids[1:max(1, valid_len - 1)]
        content_embeds = self.token_embedding(content_ids).detach()

        if content_embeds.shape[0] >= n_ctx:
            return content_embeds[:n_ctx].clone()

        pad = torch.empty(
            n_ctx - content_embeds.shape[0],
            embed_dim,
            device=device,
            dtype=dtype,
        ).normal_(std=0.02)
        return torch.cat([content_embeds, pad], dim=0)



    def _build_single_prompt(
        self,
        name_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        device = self._device()
        dtype = self._dtype()

        name_ids = name_ids.to(device)
        suffix_ids = self.suffix_token_ids.to(device)

        bos_id = torch.tensor([self.bos_token_id], device=device)
        eos_id = torch.tensor([self.eos_token_id], device=device)
        pad_id = torch.tensor([self.pad_token_id], device=device)

        bos_embed = self.token_embedding(bos_id).to(dtype)
        eos_embed = self.token_embedding(eos_id).to(dtype)
        name_embed = self.token_embedding(name_ids).to(dtype)
        suffix_embed = self.token_embedding(suffix_ids).to(dtype) if suffix_ids.numel() > 0 else torch.empty(
            0, self.ctx.shape[1], device=device, dtype=dtype
        )
        ctx_embed = self.ctx.to(dtype)

        # 预留 [BOS] + ctx + class + suffix + [EOS]
        max_name_len = max(1, self.max_length - self.n_ctx - suffix_embed.shape[0] - 2)
        name_embed = name_embed[:max_name_len]

        if self.class_token_position == "front":
            seq_embed = torch.cat(
                [bos_embed, name_embed, ctx_embed, suffix_embed, eos_embed],
                dim=0,
            )
        elif self.class_token_position == "middle":
            half = self.n_ctx // 2
            seq_embed = torch.cat(
                [bos_embed, ctx_embed[:half], name_embed, ctx_embed[half:], suffix_embed, eos_embed],
                dim=0,
            )
        else:
            # 默认 end: learned ctx 在前，类名在中，固定 suffix 在后
            seq_embed = torch.cat(
                [bos_embed, ctx_embed, name_embed, suffix_embed, eos_embed],
                dim=0,
            )

        valid_len = int(seq_embed.shape[0])
        eos_pos = valid_len - 1

        if valid_len < self.max_length:
            pad_len = self.max_length - valid_len
            pad_embed = self.token_embedding(pad_id).to(dtype).expand(pad_len, -1)
            seq_embed = torch.cat([seq_embed, pad_embed], dim=0)
        else:
            seq_embed = seq_embed[: self.max_length]
            valid_len = self.max_length
            eos_pos = self.max_length - 1

        attention_mask = torch.zeros(self.max_length, device=device, dtype=torch.long)
        attention_mask[:valid_len] = 1
        return seq_embed, attention_mask, eos_pos



    def build_prompts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
            inputs_embeds:  [C, L, D]
            attention_mask: [C, L]
            eos_positions:  [C]
        """
        all_embeds = []
        all_attention_masks = []
        all_eos_positions = []

        for name_ids in self.name_token_ids:
            seq_embed, attention_mask, eos_pos = self._build_single_prompt(name_ids)
            all_embeds.append(seq_embed)
            all_attention_masks.append(attention_mask)
            all_eos_positions.append(eos_pos)

        inputs_embeds = torch.stack(all_embeds, dim=0)
        attention_mask = torch.stack(all_attention_masks, dim=0)
        eos_positions = torch.tensor(all_eos_positions, device=inputs_embeds.device, dtype=torch.long)
        return inputs_embeds, attention_mask, eos_positions

    def forward(self) -> EncoderResult:
        inputs_embeds, attention_mask, eos_positions = self.build_prompts()
        outputs = self.text_encoder.forward_from_embeddings(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_positions=eos_positions,
            return_activations=True,
        )
        return outputs
