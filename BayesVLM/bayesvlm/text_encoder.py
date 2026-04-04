from typing import Optional, List, Union

import torch
import torch.nn as nn

# Hugging Face Transformers库
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    SiglipTextModel,
)


def _torch_load(path: str):
    return torch.load(path, map_location="cpu")


# 文本描述    - 文本投影分布  - 文本嵌入分布   - 混合文本嵌入分布


# 使用方式 可单独加载  或传入model，tokenizer
# text_encoder = CLIPTextEncoder.from_huggingface(model_name, device="cuda")
# text_features = text_encoder(["a photo of a cat", "a photo of a dog"])

# 当前文本编码器只接受文本描述列表[str]
class CLIPTextEncoder(nn.Module):
    def __init__(self, model: CLIPTextModelWithProjection, tokenizer):
        super().__init__()
        self.transformer = model.text_model
        self.text_projection = model.text_projection
        self.tokenizer = tokenizer

    @classmethod
    def from_huggingface(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        model = CLIPTextModelWithProjection.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        # cls封装  相当于调用: __init__(self, model_obj, tokenizer_obj)
        encoder = cls(model, tokenizer)
        if device is not None:
            encoder = encoder.to(device)
            encoder.device = device
        return encoder

    def load_projection_weights(self, path: str):
        state_dict = _torch_load(path)
        self.text_projection.load_state_dict(state_dict)

    def freeze_backbone(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(
        self,
        texts: Union[str, List[str]],
        return_activations: bool = False,
    ):
        text_inputs = self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(next(self.parameters()).device)
        #pooler_output：这是 Transformer 输出的精华部分。它通常对应于 [EOS]（句尾标记）位置的特征向量，代表了整句话的全局语义信息。
        outputs = self.transformer(**text_inputs)
        pooled_output = outputs.pooler_output
        text_embeds = self.text_projection(pooled_output)

        if return_activations:
            return text_embeds, pooled_output

        return text_embeds

class SiglipTextEncoder(nn.Module):
    def __init__(self, model: SiglipTextModel, tokenizer):
        super().__init__()
        self.transformer = model.text_model
        self.text_projection = model.text_model.head
        self.tokenizer = tokenizer

    @classmethod
    def from_huggingface(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        model = SiglipTextModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        encoder = cls(model, tokenizer)
        if device is not None:
            encoder = encoder.to(device)
            encoder.device = device
        return encoder

    def load_projection_weights(self, path: str):
        state_dict = _torch_load(path)
        self.text_projection.load_state_dict(state_dict)

    def freeze_backbone(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(
        self,
        texts: Union[str, List[str]],
        return_activations: bool = False,
    ):
        if isinstance(texts, str):
            texts = [texts]

        text_inputs = self.tokenizer(
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(next(self.parameters()).device)

        hidden_states = self.transformer.embeddings(**text_inputs)
        encoder_outputs = self.transformer.encoder(hidden_states)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.transformer.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[:, -1, :]
        text_embeds = self.text_projection(pooled_output)

        if return_activations:
            return text_embeds, pooled_output

        return text_embeds