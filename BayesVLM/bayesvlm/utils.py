from typing import Callable, Tuple, Literal
from bayesvlm.data.common import default_transform, siglip_transform
from bayesvlm.vlm import CLIPImageEncoder, CLIPTextEncoder, CLIP, SiglipImageEncoder, SiglipTextEncoder, SIGLIP
from bayesvlm.constants import MODEL_NAME_MAP
import os
from typing import Optional, Tuple

def get_model_type_and_size(model_str: str) -> Tuple[str, str]:
    name, size = model_str.split("-")
    return name, size

def get_image_size(model_str) -> int:
    _, _, transform_size = MODEL_NAME_MAP[model_str]
    return transform_size

def get_model_url(model_str: str) -> str:
    provider, model_id, _ = MODEL_NAME_MAP[model_str]
    return f"{provider}/{model_id}"

def get_transform(model_type: Literal['clip', 'siglip'], image_size: int) -> Callable:
    if model_type == "siglip":
        return siglip_transform(image_size)
    return default_transform(image_size)

def get_likelihood(model_type: Literal['clip', 'siglip']) -> str:
    if model_type == 'clip':
        return 'info_nce'
    return 'siglip'

def load_model(
    model_str: str,
    device: str,
    local_model_path: Optional[str] = None,
) -> Tuple[CLIPImageEncoder, CLIPTextEncoder, CLIP] | Tuple[SiglipImageEncoder, SiglipTextEncoder, SIGLIP]:
    model_type, _ = get_model_type_and_size(model_str)

    # 优先使用本地目录；否则走原来的远程/名字映射
    model_source = local_model_path if local_model_path is not None else get_model_url(model_str)

    # 只要传了本地目录，或者这个 source 本身就是本地目录，就启用离线加载
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