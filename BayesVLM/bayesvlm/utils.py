from __future__ import annotations

import os
from typing import Callable, Literal, Optional, Tuple

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.data.common import default_transform, siglip_transform
from bayesvlm.image_encoder import CLIPImageEncoder, SiglipImageEncoder
from bayesvlm.openai_clip_wrappers import load_openai_clip_rn50
from bayesvlm.text_encoder import CLIPTextEncoder, SiglipTextEncoder
from bayesvlm.vlm import CLIP, SIGLIP


def get_model_type_and_size(model_str: str) -> Tuple[str, str]:
    name, size = model_str.split("-", 1)
    return name, size


def get_image_size(model_str) -> int:
    _, _, transform_size = MODEL_NAME_MAP[model_str]
    return transform_size


def get_model_url(model_str: str) -> str:
    provider, model_id, _ = MODEL_NAME_MAP[model_str]
    return f"{provider}/{model_id}"


def get_transform(model_type: Literal["clip", "siglip"], image_size: int) -> Callable:
    if model_type == "siglip":
        return siglip_transform(image_size)
    return default_transform(image_size)


def get_likelihood(model_type: Literal["clip", "siglip"]) -> str:
    if model_type == "clip":
        return "info_nce"
    return "siglip"


def load_model(
    model_str: str,
    device: str,
    local_model_path: Optional[str] = None,
) -> Tuple[CLIPImageEncoder, CLIPTextEncoder, CLIP] | Tuple[SiglipImageEncoder, SiglipTextEncoder, SIGLIP]:
    """
    中文说明：
    - 默认仍走原来的 Hugging Face CLIP / SigLIP 路径
    - 对 clip-rn50 单独走 OpenAI CLIP RN50 包装器
    """
    model_type, model_size = get_model_type_and_size(model_str)

    # 专门给 vlm_adapter 的 OpenAI CLIP RN50
    if model_str == "clip-rn50":
        return load_openai_clip_rn50(
            device=device,
            local_model_path=local_model_path,
        )

    # 原有逻辑保持不变
    model_source = local_model_path if local_model_path is not None else get_model_url(model_str)
    local_files_only = local_model_path is not None or os.path.isdir(model_source)

    if model_type == "siglip":
        image_encoder = SiglipImageEncoder.from_huggingface(
            model_source,
            device=device,
            local_files_only=local_files_only,
        ).eval().to(device)

        text_encoder = SiglipTextEncoder.from_huggingface(
            model_source,
            device=device,
            local_files_only=local_files_only,
        ).eval().to(device)

        vlm = SIGLIP.from_huggingface(
            model_source,
            device=device,
            local_files_only=local_files_only,
        ).eval().to(device)

    elif model_type == "clip":
        image_encoder = CLIPImageEncoder.from_huggingface(
            model_source,
            device=device,
            local_files_only=local_files_only,
        ).eval().to(device)

        text_encoder = CLIPTextEncoder.from_huggingface(
            model_source,
            device=device,
            local_files_only=local_files_only,
        ).eval().to(device)

        vlm = CLIP.from_huggingface(
            model_source,
            device=device,
            local_files_only=local_files_only,
        ).eval().to(device)

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return image_encoder, text_encoder, vlm