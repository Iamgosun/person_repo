from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from bayesvlm.common import EncoderResult

try:
    import clip as openai_clip
except ImportError as e:
    raise ImportError(
        "需要安装 OpenAI CLIP 包。请先执行：pip install git+https://github.com/openai/CLIP.git"
    ) from e


def _extract_images(batch_or_images):
    if isinstance(batch_or_images, dict):
        return batch_or_images["image"]
    return batch_or_images


def _resolve_openai_clip_source(local_model_path: Optional[str] = None) -> str:
    """
    返回传给 clip.load(...) 的 source。
    支持：
    1) None -> 'RN50'
    2) 直接传 .pt 文件路径
    3) 传目录时，自动在目录下找常见文件名
    """
    if local_model_path is None:
        return "RN50"

    p = Path(local_model_path)
    if p.is_file():
        return str(p)

    if p.is_dir():
        candidates = [
            p / "RN50.pt",
            p / "rn50.pt",
            p / "clip_rn50.pt",
            p / "model.pt",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

        raise FileNotFoundError(
            f"local_model_path={local_model_path} 是目录，但未找到 RN50 checkpoint 文件。"
        )

    # 如果不是本地存在路径，就原样传入，让 clip.load 自己处理
    return local_model_path


class OpenAIClipRN50ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.visual = clip_model.visual

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_all_layers(self):
        for p in self.parameters():
            p.requires_grad = False

    def freeze_backbone(self):
        self.freeze_all_layers()

    def freeze_all_layers_except_projection(self):
        # RN50 这里没有单独暴露 HF 风格 projection，保持全冻结即可
        self.freeze_all_layers()

    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers_except_projection()

    def enable_gradients(self, k_last_layers: int = 0, enable_projection: bool = True):
        # 当前 vlm_adapter 用不到，先保留接口兼容
        if k_last_layers > 0 or enable_projection:
            for p in self.parameters():
                p.requires_grad = True

    def forward_features(self, batch_or_images):
        images = _extract_images(batch_or_images).to(self._get_device())
        return self.visual(images.type(self.clip_model.dtype))

    def forward_from_activations(
        self,
        activations: torch.Tensor,
        return_activations: bool = False,
    ):
        # 对 OpenAI CLIP 来说，visual(...) 输出已经是最终 image embedding
        image_embeds = activations
        if return_activations:
            return EncoderResult(
                embeds=image_embeds,
                activations=activations,
            )
        return image_embeds

    def forward(self, batch_or_images, return_activations: bool = False):
        activations = self.forward_features(batch_or_images)
        return self.forward_from_activations(
            activations=activations,
            return_activations=return_activations,
        )


class OpenAIClipRN50TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_all_layers(self):
        for p in self.parameters():
            p.requires_grad = False

    def freeze_backbone(self):
        self.freeze_all_layers()

    def freeze_all_layers_except_projection(self):
        self.freeze_all_layers()

    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers_except_projection()

    def enable_gradients(self, k_last_layers: int = 0, enable_projection: bool = True):
        if k_last_layers > 0 or enable_projection:
            for p in self.parameters():
                p.requires_grad = True

    def forward(self, texts: Sequence[str] | str, return_activations: bool = False):
        if isinstance(texts, str):
            texts = [texts]

        tokens = openai_clip.tokenize(list(texts)).to(self._get_device())
        text_embeds = self.clip_model.encode_text(tokens)

        if return_activations:
            return EncoderResult(
                embeds=text_embeds,
                activations=text_embeds,
            )
        return text_embeds


class OpenAIClipRN50VLM(nn.Module):
    source_projection_has_bias = False
    target_projection_has_bias = False

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.logit_bias = None

    @property
    def device(self):
        return self.logit_scale.device

    def forward(
        self,
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        map_estimate: bool = False,
    ):
        del map_estimate

        source_embeds = source_embeds / source_embeds.norm(dim=-1, keepdim=True)
        target_embeds = target_embeds / target_embeds.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(source_embeds, target_embeds.t())
        similarity = similarity * self.logit_scale.exp()
        return similarity


def load_openai_clip_rn50(
    device: str,
    local_model_path: Optional[str] = None,
):
    """
    返回与你当前项目兼容的三件套：
    - image_encoder
    - text_encoder
    - vlm
    """
    source = _resolve_openai_clip_source(local_model_path)
    clip_model, _ = openai_clip.load(source, device=device, jit=False)

    clip_model = clip_model.eval()

    image_encoder = OpenAIClipRN50ImageEncoder(clip_model).eval().to(device)
    text_encoder = OpenAIClipRN50TextEncoder(clip_model).eval().to(device)
    vlm = OpenAIClipRN50VLM(clip_model).eval().to(device)

    return image_encoder, text_encoder, vlm