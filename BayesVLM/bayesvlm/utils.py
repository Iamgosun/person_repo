# bayesvlm/utils.py
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


def _resolve_existing_local_path(path_str: Optional[str]) -> Optional[str]:
    """
    中文说明：
    - 只把“真实存在”的目录/文件当作本地路径。
    - 不存在时返回 None，交给后续逻辑决定是否回退到 HF repo id。
    """
    if path_str is None:
        return None

    raw = str(path_str).strip()
    if raw == "":
        return None

    expanded = os.path.abspath(os.path.expanduser(raw))
    if os.path.exists(expanded):
        return expanded

    return None


def _looks_like_hf_repo_id(path_str: str) -> bool:
    """
    中文说明：
    尽量保守地识别 Hugging Face repo id，例如：
        openai/clip-vit-base-patch16

    这里故意不把 ./models/xxx 这种相对路径识别成 repo id。
    """
    raw = str(path_str).strip()

    if raw == "":
        return False

    if raw.startswith(("/", "./", "../", "~")):
        return False

    if "\\" in raw:
        return False

    parts = raw.split("/")
    if len(parts) != 2:
        return False

    # 常见本地相对目录前缀，避免误判
    if parts[0].lower() in {"models", "datasets", "cache", "output"}:
        return False

    return all(part != "" for part in parts)


def _resolve_model_source(
    model_str: str,
    local_model_path: Optional[str],
) -> tuple[str, bool]:
    """
    中文说明：
    返回：
        model_source, local_files_only

    解析优先级：
    1. local_model_path 指向真实存在的本地目录 -> 走本地
    2. local_model_path 是显式 HF repo id -> 走远端 / 本地缓存
    3. 其他情况 -> 回退到 MODEL_NAME_MAP 对应 repo id
    """
    default_source = get_model_url(model_str)

    if local_model_path is None or str(local_model_path).strip() == "":
        print(f"[load_model] 未提供 local_model_path，使用默认模型源：{default_source}")
        return default_source, False

    raw = str(local_model_path).strip()
    existing_local_path = _resolve_existing_local_path(raw)
    if existing_local_path is not None:
        print(f"[load_model] 使用本地模型目录：{existing_local_path}")
        return existing_local_path, True

    if _looks_like_hf_repo_id(raw):
        print(f"[load_model] 检测到 Hugging Face repo id：{raw}")
        return raw, False

    print(
        f"[load_model] local_model_path={raw} 不存在，"
        f"回退到 MODEL_NAME_MAP 对应模型源：{default_source}"
    )
    return default_source, False


def load_model(
    model_str: str,
    device: str,
    local_model_path: Optional[str] = None,
) -> Tuple[CLIPImageEncoder, CLIPTextEncoder, CLIP] | Tuple[SiglipImageEncoder, SiglipTextEncoder, SIGLIP]:
    """
    中文说明：
    - 默认走 Hugging Face CLIP / SigLIP 路径
    - 对 clip-rn50 单独走 OpenAI CLIP RN50 包装器
    - 新增支持：
        1) local_model_path 为真实本地目录
        2) local_model_path 为 HF repo id（尚未手动下载时非常有用）
        3) local_model_path 无效时自动回退到 MODEL_NAME_MAP 对应 repo id
    """
    model_type, _ = get_model_type_and_size(model_str)

    # 专门给 vlm_adapter 的 OpenAI CLIP RN50
    if model_str == "clip-rn50":
        return load_openai_clip_rn50(
            device=device,
            local_model_path=local_model_path,
        )

    model_source, local_files_only = _resolve_model_source(
        model_str=model_str,
        local_model_path=local_model_path,
    )

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