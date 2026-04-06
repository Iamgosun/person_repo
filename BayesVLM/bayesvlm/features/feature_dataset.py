from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from bayesvlm.features.image_cache import ImageFeatureBundle


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