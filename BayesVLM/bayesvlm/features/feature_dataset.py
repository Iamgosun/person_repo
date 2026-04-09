from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from bayesvlm.features.image_cache import ImageFeatureBundle


def build_random_repeated_feature_loader(
    bundle: ImageFeatureBundle,
    repeats: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
):
    ds = RandomRepeatedCachedFeatureDataset(bundle, repeats=repeats)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=torch.cuda.is_available(),
    )

class RandomRepeatedCachedFeatureDataset(Dataset):
    """
    把 flat 的增强缓存 [repeats * base_n, ...]
    重新视作 [repeats, base_n, ...]，
    每次 __getitem__(idx) 时随机采样一个增强视角。
    这样每个 epoch 每个 base sample 只贡献一次，
    行为更接近 BayesAdapter。
    """

    def __init__(self, bundle: ImageFeatureBundle, repeats: int):
        self.bundle = bundle
        self.repeats = int(repeats)

        total_n = int(bundle.class_ids.shape[0])
        if total_n % self.repeats != 0:
            raise ValueError(
                f"Augmented bundle size {total_n} is not divisible by repeats={self.repeats}"
            )

        self.base_n = total_n // self.repeats

        self.embeds = bundle.outputs.embeds.view(
            self.repeats, self.base_n, *bundle.outputs.embeds.shape[1:]
        )

        self.activations = None
        if bundle.outputs.activations is not None:
            self.activations = bundle.outputs.activations.view(
                self.repeats, self.base_n, *bundle.outputs.activations.shape[1:]
            )

        self.residuals = None
        if bundle.outputs.residuals is not None:
            self.residuals = bundle.outputs.residuals.view(
                self.repeats, self.base_n, *bundle.outputs.residuals.shape[1:]
            )

        # 类别、路径、key 只保留 base 样本那一份
        self.class_ids = bundle.class_ids[: self.base_n]
        self.sample_keys = bundle.sample_keys[: self.base_n]
        self.image_paths = None if bundle.image_paths is None else bundle.image_paths[: self.base_n]

    def __len__(self) -> int:
        return self.base_n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rep = torch.randint(low=0, high=self.repeats, size=(1,)).item()

        row = {
            "image_embeds": self.embeds[rep, idx],
            "class_id": self.class_ids[idx],
            "image_id": idx,
            "sample_key": self.sample_keys[idx],
        }

        if self.activations is not None:
            row["activations"] = self.activations[rep, idx]
        if self.residuals is not None:
            row["residuals"] = self.residuals[rep, idx]
        if self.image_paths is not None:
            row["image_path"] = self.image_paths[idx]

        return row
class CachedImageFeatureDataset(Dataset):
    def __init__(self, bundle: ImageFeatureBundle):
        self.bundle = bundle

    def __len__(self):
        return self.bundle.class_ids.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = {
            "image_embeds": self.bundle.outputs.embeds[idx],
            "activations": self.bundle.outputs.activations[idx],
            "residuals": self.bundle.outputs.residuals[idx],
            "class_id": self.bundle.class_ids[idx],
            "image_id": idx,
            "sample_key": self.bundle.sample_keys[idx],
        }
        if self.bundle.image_paths is not None:
            row["image_path"] = self.bundle.image_paths[idx]
        return row


def build_feature_loader(
    bundle: ImageFeatureBundle,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
):
    ds = CachedImageFeatureDataset(bundle)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=torch.cuda.is_available(),
    )