from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence
import csv

import numpy as np
import torch
import pytorch_lightning as L
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from .common import default_collate_fn, default_transform


def _read_split_csv(csv_path: Path):
    entries = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [x.strip() for x in row if x is not None and str(x).strip() != ""]
            if not row:
                continue

            lower = [x.lower() for x in row]
            header_tokens = {
                "path", "image", "images", "file", "filename",
                "class", "class_name", "label", "id", "image_id", "stem"
            }
            if all(x in header_tokens for x in lower):
                continue

            if len(row) == 1:
                token = row[0]
            else:
                # 常见情况:
                # 1) apple_pie/1005649
                # 2) apple_pie,1005649
                # 3) apple_pie,1005649.jpg
                if "/" in row[0] or "\\" in row[0]:
                    token = row[0]
                else:
                    token = f"{row[0]}/{row[1]}"

            token = token.strip().strip('"').strip("'")
            if not token:
                continue

            rel_path = Path(token)
            if rel_path.suffix == "":
                rel_path = rel_path.with_suffix(".jpg")

            entries.append(rel_path)

    return entries


def _build_samples_from_split(images_root: Path, split_entries, class_to_idx):
    samples = []
    for rel_path in split_entries:
        image_path = images_root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file referenced by split: {image_path}")

        class_name = rel_path.parts[0]
        if class_name not in class_to_idx:
            raise ValueError(
                f"Class '{class_name}' from split file not found under images root {images_root}"
            )

        class_id = class_to_idx[class_name]
        samples.append((image_path, class_id))
    return samples


def _sample_few_shot_indices(samples, shots_per_class: int, seed: int):
    class_to_indices = defaultdict(list)
    for idx, (_, class_id) in enumerate(samples):
        class_to_indices[class_id].append(idx)

    rng = np.random.default_rng(seed)
    selected = []
    for class_id in sorted(class_to_indices.keys()):
        indices = class_to_indices[class_id]
        if len(indices) < shots_per_class:
            raise ValueError(
                f"Class {class_id} only has {len(indices)} samples, "
                f"but shots_per_class={shots_per_class}."
            )
        chosen = rng.choice(indices, size=shots_per_class, replace=False).tolist()
        selected.extend(chosen)

    return selected


class Food101LocalDataset(Dataset):
    def __init__(self, samples, label_names, text_prompt: str, transform=None):
        self._samples = samples
        self._label_names = label_names
        self._text_prompt = text_prompt
        self._transform = transform

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        image_path, class_id = self._samples[idx]

        image = Image.open(image_path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)

        text = self._text_prompt.format(class_name=self._label_names[class_id])

        return {
            "image": image,
            "text": text,
            "class_id": class_id,
            "image_id": idx,
            "image_path": str(image_path),
        }


class Food101DataModule(L.LightningDataModule):
    DATASET_SUBDIR = "food101"

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        text_prompt: str = "An image of a {class_name}",
        train_transform=None,
        test_transform=None,
        shuffle_train: bool = True,
        subset_indices: Optional[Sequence[int]] = None,
        shots_per_class: int = 10,
        use_few_shot: bool = False,
        few_shot_sample_seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.text_prompt = text_prompt
        self.train_transform = train_transform or default_transform(image_size=224)
        self.test_transform = test_transform or default_transform(image_size=224)
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices

        self.use_few_shot = use_few_shot
        self.shots_per_class = shots_per_class
        self.few_shot_sample_seed = few_shot_sample_seed

        self.label_names = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str = None):
        dataset_root = self.data_dir / "food-101"
        images_root = dataset_root / "images"
        meta_root = dataset_root / "meta"

        train_csv = meta_root / "train.csv"
        test_csv = meta_root / "test.csv"

        if not images_root.is_dir():
            raise FileNotFoundError(f"Food101 images root not found: {images_root}")
        if not train_csv.exists():
            raise FileNotFoundError(f"Food101 split file not found: {train_csv}")
        if not test_csv.exists():
            raise FileNotFoundError(f"Food101 split file not found: {test_csv}")

        folder = ImageFolder(images_root)
        self.label_names = folder.classes
        class_to_idx = folder.class_to_idx

        official_train_entries = _read_split_csv(train_csv)
        official_test_entries = _read_split_csv(test_csv)

        official_train_samples = _build_samples_from_split(
            images_root, official_train_entries, class_to_idx
        )
        official_test_samples = _build_samples_from_split(
            images_root, official_test_entries, class_to_idx
        )

        # 保持源码逻辑: 官方 train -> 再切出 val
        indices = list(range(len(official_train_samples)))
        indices_train, indices_val = train_test_split(
            indices, test_size=0.2, random_state=0
        )

        train_samples = [official_train_samples[i] for i in indices_train]
        val_samples = [official_train_samples[i] for i in indices_val]
        test_samples = official_test_samples

        train_ds = Food101LocalDataset(
            train_samples,
            label_names=self.label_names,
            text_prompt=self.text_prompt,
            transform=self.train_transform,
        )

        if self.use_few_shot:
            few_shot_indices = _sample_few_shot_indices(
                train_samples,
                shots_per_class=self.shots_per_class,
                seed=self.few_shot_sample_seed,
            )
            train_ds = Subset(train_ds, few_shot_indices)

        if self.subset_indices is not None:
            train_ds = Subset(train_ds, self.subset_indices)

        self.train_ds = train_ds
        self.val_ds = Food101LocalDataset(
            val_samples,
            label_names=self.label_names,
            text_prompt=self.text_prompt,
            transform=self.test_transform,
        )
        self.test_ds = Food101LocalDataset(
            test_samples,
            label_names=self.label_names,
            text_prompt=self.text_prompt,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=default_collate_fn,
            shuffle=self.shuffle_train,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=default_collate_fn,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=default_collate_fn,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
        )

    @property
    def class_prompts(self):
        if self.label_names is None:
            raise RuntimeError("Call setup() before accessing class_prompts.")
        return [self.text_prompt.format(class_name=name) for name in self.label_names]