from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch

#from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT
from bayesvlm.hessians import KroneckerFactorizedCovariance
from bayesvlm.text_encoder import CLIPTextEncoder

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

TextEncoderType = Union[CLIPTextEncoder]


@dataclass
class GaussianBatch:
    """
    一批高斯分布

    mean: [N, D]
    covariance: [N, D, D]
    """
    mean: torch.Tensor
    covariance: torch.Tensor


@dataclass
class ClassTextDistributions:
    """
    类别级文本嵌入分布

    class_names:
        类别名列表，长度为 C

    mean:
        每个类别最终混合后的高斯均值
        [C, D]

    covariance:
        每个类别最终混合后的高斯协方差
        [C, D, D]

    description_texts:
        每个类别经过 template 展开后的文本描述
        list[list[str]]，长度为 C，每项长度为 T

    description_means:
        每个类别下，每个模板文本描述对应的高斯均值
        list[[T, D]]

    description_covariances:
        每个类别下，每个模板文本描述对应的高斯协方差
        list[[T, D, D]]
    """
    class_names: List[str]
    mean: torch.Tensor
    covariance: torch.Tensor
    description_texts: List[List[str]]
    description_means: List[torch.Tensor]
    description_covariances: List[torch.Tensor]


def _build_templates(dataset_name: str) -> List[str]:
    if dataset_name == "ImageNet":
        IMAGENET_TEMPLATES_SELECT=None
        templates = list(IMAGENET_TEMPLATES_SELECT)
    else:
        templates = []

    if dataset_name in CUSTOM_TEMPLATES:
        templates.append(CUSTOM_TEMPLATES[dataset_name])

    if len(templates) == 0:
        templates = [ "a photo of a {}."]
        #raise ValueError(f"No template found for dataset: {dataset_name}")

    return templates



@torch.no_grad()
def compute_text_posterior_gaussians(
    text_encoder: TextEncoderType,
    covariance: KroneckerFactorizedCovariance,
    prompts: Sequence[str],
    device: str | torch.device = "cpu",
) -> GaussianBatch:
    """
    计算每个 prompt / 文本描述 的文本嵌入后验高斯

    prompts: [N]
    返回:
        mean: [N, D]
        covariance: [N, D, D]
    """
    if len(prompts) == 0:
        raise ValueError("prompts must not be empty.")

    device = torch.device(device)

    B_inv = covariance.B_inv.to(device)
    A_inv = covariance.A_inv.to(device)

    has_bias = getattr(text_encoder.text_projection, "bias", None) is not None

    text_embeds, activations = text_encoder(
        list(prompts),
        return_activations=True,
    )

    text_embeds = text_embeds.to(device)
    activations = activations.to(device)

    if has_bias:
        ones = torch.ones(
            activations.shape[0],
            1,
            device=device,
            dtype=activations.dtype,
        )
        activations = torch.cat([activations, ones], dim=-1)

    # scalar[n] = a_n^T A_inv a_n
    scalar = torch.einsum("bi,ij,bj->b", activations, A_inv, activations)

    # cov[n] = scalar[n] * B_inv
    cov = scalar[:, None, None] * B_inv.unsqueeze(0)

    return GaussianBatch(
        mean=text_embeds,
        covariance=cov,
    )


def aggregate_gaussians(
    means: torch.Tensor,
    covariances: torch.Tensor,
    jitter: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将多个高斯分布做 moment matching，混合成一个单高斯

    means: [N, D]
    covariances: [N, D, D]

    返回:
        mixed_mean: [D]
        mixed_covariance: [D, D]
    """
    if means.ndim != 2:
        raise ValueError(f"means should be [N, D], got {means.shape}")
    if covariances.ndim != 3:
        raise ValueError(f"covariances should be [N, D, D], got {covariances.shape}")
    if means.shape[0] != covariances.shape[0]:
        raise ValueError("Number of means and covariances must match.")
    if means.shape[0] == 0:
        raise ValueError("Cannot aggregate an empty Gaussian set.")

    mixed_mean = means.mean(dim=0)

    second_moment = (
        covariances
        + means.unsqueeze(-1) @ means.unsqueeze(-2)
    ).mean(dim=0)

    mixed_covariance = (
        second_moment
        - mixed_mean.unsqueeze(-1) @ mixed_mean.unsqueeze(-2)
    )

    if jitter > 0:
        eye = torch.eye(
            mixed_covariance.shape[-1],
            device=mixed_covariance.device,
            dtype=mixed_covariance.dtype,
        )
        mixed_covariance = mixed_covariance + jitter * eye

    return mixed_mean, mixed_covariance


@torch.no_grad()
def get_class_text_distributions(
    dataset_name: str,
    class_names: Sequence[str],
    text_encoder: TextEncoderType,
    covariance: KroneckerFactorizedCovariance,

    device: str | torch.device = "cpu",
    jitter: float = 1e-5,
) -> ClassTextDistributions:
    """
    类别级文本嵌入分布构建函数

    正确逻辑:
        类别名
          -> 用多个 template 生成多个文本描述
          -> 每个文本描述编码成一个高斯
          -> 把这些文本描述高斯混合
          -> 得到类别高斯

    输入:
        dataset_name:
            数据集名，用于选择模板

        class_names:
            类别名列表，例如:
            ["Persian_cat", "Siamese_cat", "golden_retriever"]

        text_encoder:
            CLIPTextEncoder / SiglipTextEncoder

        covariance:
            文本投影层协方差

        device:
            设备

        jitter:
            协方差稳定项

    返回:
        ClassTextDistributions
    """
    if len(class_names) == 0:
        raise ValueError("class_names must not be empty.")

    device = torch.device(device)

    text_encoder = text_encoder.to(device)
    text_encoder.device = device


    templates = _build_templates(dataset_name)

    all_class_names: List[str] = []
    all_class_means: List[torch.Tensor] = []
    all_class_covariances: List[torch.Tensor] = []

    all_description_texts: List[List[str]] = []
    all_description_means: List[torch.Tensor] = []
    all_description_covariances: List[torch.Tensor] = []

    for class_name in class_names:
        # 一个类别名，经多个 template 展开成多个文本描述
        description_texts = [
            template.format(class_name) for template in templates
        ]

        # 每个文本描述 -> 一个高斯
        description_batch = compute_text_posterior_gaussians(
            text_encoder=text_encoder,
            covariance=covariance,
            prompts=description_texts,
            device=device,
        )

        # 多个文本描述高斯 -> 混合成一个类别高斯
        class_mean, class_covariance = aggregate_gaussians(
            means=description_batch.mean,
            covariances=description_batch.covariance,
            jitter=jitter,
        )

        all_class_names.append(class_name)
        all_class_means.append(class_mean)
        all_class_covariances.append(class_covariance)

        all_description_texts.append(description_texts)
        all_description_means.append(description_batch.mean)
        all_description_covariances.append(description_batch.covariance)

    return ClassTextDistributions(
        class_names=all_class_names,
        mean=torch.stack(all_class_means, dim=0),               # [C, D]
        covariance=torch.stack(all_class_covariances, dim=0),   # [C, D, D]
        description_texts=all_description_texts,
        description_means=all_description_means,                # list[[T, D]]
        description_covariances=all_description_covariances,    # list[[T, D, D]]
    )