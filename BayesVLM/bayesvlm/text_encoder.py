from __future__ import annotations

from typing import Optional, Sequence

import torch
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    SiglipTextModel,
)

from bayesvlm.common import EncoderResult, get_projection_dim, torch_load_cpu


def _make_clIP_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    中文说明：
    CLIP 文本 transformer 使用 causal mask。
    这里手动构造 [B, 1, L, L] 的 additive mask。
    """
    mask = torch.full(
        (seq_len, seq_len),
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    return mask



def _make_padding_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    中文说明：
    把 [B, L] 的 0/1 mask 转成 CLIP encoder 需要的 [B, 1, L, L] additive mask。
    其中：
    - 1 表示有效 token
    - 0 表示 padding token
    """
    batch_size, seq_len = attention_mask.shape

    # 先变成 [B, 1, 1, L]
    mask = 1.0 - attention_mask[:, None, None, :].to(dtype)

    # 再扩成 [B, 1, L, L]
    mask = mask.expand(batch_size, 1, seq_len, seq_len)

    # 转成 additive mask，padding 位置给极小值
    mask = mask * torch.finfo(dtype).min
    return mask




class CLIPTextEncoder(torch.nn.Module):
    """
    中文说明：
    1. 这是从 vlm.py 中解耦出来的 CLIP 文本编码器。
    2. 除了原来的字符串输入 forward，还额外支持：
       - tokenize()
       - get_token_embedding_layer()
       - forward_from_embeddings()
    3. 这些接口是后续做 CoOp soft prompt 的关键。
    """

    def __init__(
        self,
        text_model: CLIPTextModelWithProjection,
        tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self.text_encoder = text_model.text_model
        self.text_projection = text_model.text_projection
        self.tokenizer = tokenizer
        self.device = getattr(text_model, "device", None)

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        if projection_dim is None:
            projection_dim = get_projection_dim(
                model_name,
                local_files_only=local_files_only,
            )

        text_model = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            projection_dim=projection_dim,
            local_files_only=local_files_only,
        )
        model = cls(text_model, tokenizer)
        model = model.to(device) if device is not None else model
        model.device = device
        return model

    def save_projection_weights(self, path: str):
        torch.save(self.text_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.text_projection.load_state_dict(state_dict)
            return

        if path is None:
            raise ValueError("Either path or state_dict must be provided.")

        self.text_projection.load_state_dict(torch_load_cpu(path))

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """
        中文说明：
        只冻结 text backbone，不动 projection。
        """
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def freeze_all_layers_except_projection(self):
        self.freeze_all_layers()
        for param in self.text_projection.parameters():
            param.requires_grad = True

    # 兼容旧代码里的拼写
    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers_except_projection()

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.text_projection.train()
            for param in self.text_projection.parameters():
                param.requires_grad = True

        if k_last_layers > 0:
            for layer in self.text_encoder.encoder.layers[-k_last_layers:]:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True

    def tokenize(
        self,
        texts: Sequence[str],
        padding=True,
        truncation=True,
        return_tensors: str = "pt",
    ):
        return self.tokenizer(
            text=list(texts),
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        ).to(self._get_device())

    def get_token_embedding_layer(self):
        return self.text_encoder.embeddings.token_embedding

    def forward_tokenized(self, tokenized, return_activations=False):
        text_outputs = self.text_encoder(**tokenized)
        text_pooled_output = text_outputs[1]
        text_embeds = self.text_projection(text_pooled_output)

        if return_activations:
            return EncoderResult(
                embeds=text_embeds,
                activations=text_pooled_output,
            )

        return text_embeds

    def forward_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_positions: Optional[torch.Tensor] = None,
        return_activations: bool = False,
    ):
        """
        中文说明：
        直接从 embedding 序列做前向。
        这是实现 CoOp / soft prompt 的关键接口。
        """
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        batch_size, seq_len, _ = inputs_embeds.shape

        if eos_positions is None:
            eos_positions = attention_mask.sum(dim=-1) - 1
        eos_positions = eos_positions.to(device=device, dtype=torch.long)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        position_embeds = self.text_encoder.embeddings.position_embedding(position_ids)

        hidden_states = inputs_embeds + position_embeds

        causal_attention_mask = _make_clIP_causal_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            device=device,
        )
        padding_attention_mask = _make_padding_attention_mask(
            attention_mask=attention_mask,
            dtype=dtype,
        )

        encoder_outputs = self.text_encoder.encoder(
            inputs_embeds=hidden_states,
            attention_mask=padding_attention_mask,
            causal_attention_mask=causal_attention_mask,
            return_dict=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.text_encoder.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[
            torch.arange(batch_size, device=device),
            eos_positions,
        ]
        text_embeds = self.text_projection(pooled_output)

        if return_activations:
            return EncoderResult(
                embeds=text_embeds,
                activations=pooled_output,
            )

        return text_embeds


    def forward(self, texts, return_activations=False):
        """
        只接受：
        1) list[str] / tuple[str]
        不再接受 batch dict
        """
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenize(
            texts=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return self.forward_tokenized(tokenized, return_activations=return_activations)
















class SiglipTextEncoder(torch.nn.Module):
    """
    中文说明：
    1. 这是从 vlm.py 中解耦出来的 SigLIP 文本编码器。
    2. 同样补了 tokenize / embedding-level forward 接口。
    3. 这样后续如果你想把 CoOp 或别的 prompt 方法迁到 SigLIP，也更顺手。
    """

    def __init__(
        self,
        model: SiglipTextModel,
        tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self._siglip_text_transformer = model.text_model
        self.text_projection = model.text_model.head
        self.tokenizer = tokenizer
        self.device = getattr(model, "device", None)

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        model = SiglipTextModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model = cls(model, tokenizer)
        model = model.to(device) if device is not None else model
        model.device = device
        return model

    def save_projection_weights(self, path: str):
        torch.save(self.text_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.text_projection.load_state_dict(state_dict)
            return

        if path is None:
            raise ValueError("Either path or state_dict must be provided.")

        self.text_projection.load_state_dict(torch_load_cpu(path))

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        for param in self._siglip_text_transformer.parameters():
            param.requires_grad = False

    def freeze_all_layers_except_projection(self):
        self.freeze_all_layers()
        for param in self.text_projection.parameters():
            param.requires_grad = True

    # 兼容旧代码里的拼写
    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers_except_projection()

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.text_projection.train()
            for param in self.text_projection.parameters():
                param.requires_grad = True

        if k_last_layers > 0:
            for layer in self._siglip_text_transformer.encoder.layers[-k_last_layers:]:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True

    def tokenize(
        self,
        texts: Sequence[str],
        padding="max_length",
        truncation=True,
        return_tensors: str = "pt",
    ):
        return self.tokenizer(
            text=list(texts),
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        ).to(self._get_device())

    def get_token_embedding_layer(self):
        embeddings = self._siglip_text_transformer.embeddings
        if hasattr(embeddings, "token_embedding"):
            return embeddings.token_embedding
        if hasattr(embeddings, "word_embeddings"):
            return embeddings.word_embeddings
        raise AttributeError("Cannot find token embedding layer in SigLIP text embeddings.")

    def _get_position_embedding(self, position_ids: torch.Tensor):
        embeddings = self._siglip_text_transformer.embeddings
        if hasattr(embeddings, "position_embedding"):
            return embeddings.position_embedding(position_ids)
        if hasattr(embeddings, "position_embeddings"):
            return embeddings.position_embeddings(position_ids)
        return 0.0

    def forward_tokenized(self, tokenized, return_activations=False):
        attention_mask = tokenized.get("attention_mask", None)

        hidden_states = self._siglip_text_transformer.embeddings(**tokenized)
        try:
            encoder_outputs = self._siglip_text_transformer.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
        except TypeError:
            encoder_outputs = self._siglip_text_transformer.encoder(hidden_states)

        last_hidden_state = encoder_outputs[0] if isinstance(encoder_outputs, tuple) else encoder_outputs.last_hidden_state
        last_hidden_state = self._siglip_text_transformer.final_layer_norm(last_hidden_state)

        if attention_mask is None:
            pooled_output = last_hidden_state[:, -1, :]
        else:
            last_index = attention_mask.sum(dim=-1) - 1
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.size(0), device=last_hidden_state.device),
                last_index,
            ]

        text_embeds = self.text_projection(pooled_output)

        if return_activations:
            return EncoderResult(
                embeds=text_embeds,
                activations=pooled_output,
            )

        return text_embeds

    def forward_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_positions: Optional[torch.Tensor] = None,
        return_activations: bool = False,
    ):
        device = inputs_embeds.device
        batch_size, seq_len, _ = inputs_embeds.shape

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        position_embeds = self._get_position_embedding(position_ids)
        hidden_states = inputs_embeds + position_embeds

        try:
            encoder_outputs = self._siglip_text_transformer.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
        except TypeError:
            encoder_outputs = self._siglip_text_transformer.encoder(hidden_states)

        last_hidden_state = encoder_outputs[0] if isinstance(encoder_outputs, tuple) else encoder_outputs.last_hidden_state
        last_hidden_state = self._siglip_text_transformer.final_layer_norm(last_hidden_state)

        if eos_positions is None:
            eos_positions = attention_mask.sum(dim=-1) - 1

        pooled_output = last_hidden_state[
            torch.arange(batch_size, device=device),
            eos_positions.to(device=device, dtype=torch.long),
        ]
        text_embeds = self.text_projection(pooled_output)

        if return_activations:
            return EncoderResult(
                embeds=text_embeds,
                activations=pooled_output,
            )

        return text_embeds

    def forward(self, batch, return_activations=False):
        texts = batch["text"] if isinstance(batch, dict) else batch
        tokenized = self.tokenize(
            texts=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return self.forward_tokenized(tokenized, return_activations=return_activations)