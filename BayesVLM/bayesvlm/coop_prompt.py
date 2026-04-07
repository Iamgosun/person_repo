from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from bayesvlm.common import EncoderResult
from bayesvlm.text_encoder import CLIPTextEncoder, SiglipTextEncoder


class CoOpPromptLearner(nn.Module):
    """
    更接近标准 CoOp 的 PromptLearner

    关键点：
    1. 若给定 ctx_init，则 n_ctx 直接按 ctx_init 的词数重置，不再随机补齐
    2. 先构造完整 prompt，再缓存 token_prefix / token_suffix
    3. forward 时只替换中间的 ctx，其余 token 来自完整 prompt 的真实 tokenization
    4. 支持 generic context 和 class-specific context (CSC)
    """

    def __init__(
        self,
        class_names: Sequence[str],
        text_encoder: CLIPTextEncoder | SiglipTextEncoder,
        n_ctx: int = 16,
        ctx_init: str = "",
        csc: bool = False,
        class_token_position: str = "end",
    ):
        super().__init__()

        self.class_names = [str(name).replace("_", " ") for name in class_names]
        self.text_encoder = text_encoder
        self.tokenizer = text_encoder.tokenizer
        self.token_embedding = text_encoder.get_token_embedding_layer()

        self.n_cls = len(self.class_names)
        self.class_token_position = class_token_position
        self.csc = bool(csc)

        embed_weight = self.token_embedding.weight
        dtype = embed_weight.dtype
        device = embed_weight.device
        ctx_dim = embed_weight.shape[1]

        if ctx_init and len(ctx_init.strip()) > 0:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))

            prompt = self.tokenizer(
                ctx_init,
                return_tensors="pt",
                truncation=True,
            )
            input_ids = prompt["input_ids"][0].to(device)

            with torch.no_grad():
                embedding = self.token_embedding(input_ids).to(dtype)

            ctx_vectors = embedding[1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            if self.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(
                    self.n_cls, n_ctx, ctx_dim, device=device, dtype=dtype
                )
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(
                    n_ctx, ctx_dim, device=device, dtype=dtype
                )

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.n_ctx = int(n_ctx)

        prompts = [prompt_prefix + " " + name + "." for name in self.class_names]

        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            embedding = self.token_embedding(input_ids).to(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])

        self.register_buffer("prompt_input_ids", input_ids)
        self.register_buffer("prompt_attention_mask", attention_mask)

        self.name_lens = []
        for name in self.class_names:
            ids = self.tokenizer(
                name,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            self.name_lens.append(int(ids.shape[0]))

    def _ctx_per_class(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return ctx

    def build_prompts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self._ctx_per_class()
        prefix = self.token_prefix
        suffix = self.token_suffix
        attention_mask = self.prompt_attention_mask

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]

                prompt_i = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                out.append(prompt_i)
            prompts = torch.cat(out, dim=0)

        elif self.class_token_position == "front":
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1]

                prompt_i = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                out.append(prompt_i)
            prompts = torch.cat(out, dim=0)

        else:
            raise ValueError(f"Invalid class_token_position: {self.class_token_position}")

        eos_positions = attention_mask.sum(dim=-1) - 1
        return prompts, attention_mask, eos_positions

    def forward(self) -> EncoderResult:
        inputs_embeds, attention_mask, eos_positions = self.build_prompts()
        outputs = self.text_encoder.forward_from_embeddings(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_positions=eos_positions,
            return_activations=True,
        )
        return outputs