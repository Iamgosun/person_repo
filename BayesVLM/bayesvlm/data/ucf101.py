import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytorch_lightning as L
import torch
from PIL import Image
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset

from .common import default_collate_fn, default_transform


class UCF101Dataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        split_info: list,
        text_prompt: str,
        transform=None,
        use_few_shot: bool = False,
        shots_per_class: int = 5,
        few_shot_sample_seed: int = 0,
    ):
        self._data_dir = Path(image_dir)
        self._split_info = split_info
        self._text_prompt = text_prompt
        self._transform = transform

        idx_to_classname = {x[1]: x[2] for x in split_info}
        self._label_names = [idx_to_classname[x] for x in sorted(idx_to_classname.keys())]

        self.use_few_shot = use_few_shot
        if self.use_few_shot:
            self.shots_per_class = shots_per_class
            self.few_shot_sample_seed = few_shot_sample_seed

            class_index = defaultdict(list)
            for i in range(len(self._split_info)):
                class_id = self._split_info[i][1]
                class_index[class_id].append(i)

            rng = np.random.default_rng(self.few_shot_sample_seed)
            selected_data = []
            for indices in class_index.values():
                if len(indices) < self.shots_per_class:
                    raise ValueError(
                        f"A class only has {len(indices)} samples, "
                        f"but shots_per_class={self.shots_per_class}."
                    )
                selected_data.extend(
                    rng.choice(indices, self.shots_per_class, replace=False).tolist()
                )
            self.selected_data = selected_data

    def __len__(self):
        return len(self._split_info)

    def __getitem__(self, idx):
        rel_path, class_id, class_name = self._split_info[idx]
        image_path = self._data_dir / rel_path

        image = Image.open(image_path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)

        text = self._text_prompt.format(class_name=class_name)

        return dict(
            image=image,
            text=text,
            class_id=class_id,
            image_id=idx,
            image_path=str(image_path),
        )


class UCF101DataModule(L.LightningDataModule):
    DATASET_SUBDIR = "ucf101"

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        text_prompt: str = "An image of a {class_name}",
        train_transform=default_transform(image_size=244),
        test_transform=default_transform(image_size=244),
        shuffle_train: bool = True,
        subset_indices: Sequence[int] = None,
        shots_per_class: int = 10,
        use_few_shot: bool = False,
        few_shot_sample_seed: int = 42,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices

        self.use_few_shot = use_few_shot
        self.shots_per_class = shots_per_class
        self.few_shot_sample_seed = few_shot_sample_seed

    def setup(self, stage: str = None):
        splits_file = self.data_dir / "split_zhou_UCF101.json"
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Missing split file: {splits_file}. "
                "UCF101 must keep the original split_zhou_UCF101.json logic."
            )

        with open(splits_file) as f:
            splits_info = json.load(f)

        image_dir = self.data_dir / "UCF-101-midframes"

        if self.use_few_shot:
            self.train_ds = UCF101Dataset(
                image_dir=image_dir,
                split_info=splits_info["train"],
                text_prompt=self.text_prompt,
                transform=self.train_transform,
                use_few_shot=True,
                shots_per_class=self.shots_per_class,
                few_shot_sample_seed=self.few_shot_sample_seed,
            )
            self.train_ds = Subset(self.train_ds, self.train_ds.selected_data)
        else:
            self.train_ds = UCF101Dataset(
                image_dir=image_dir,
                split_info=splits_info["train"],
                text_prompt=self.text_prompt,
                transform=self.train_transform,
            )

        if self.subset_indices is not None:
            self.train_ds = Subset(self.train_ds, self.subset_indices)

        self.val_ds = UCF101Dataset(
            image_dir=image_dir,
            split_info=splits_info["val"],
            text_prompt=self.text_prompt,
            transform=self.test_transform,
        )

        self.test_ds = UCF101Dataset(
            image_dir=image_dir,
            split_info=splits_info["test"],
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
        if self.use_few_shot:
            return [self.text_prompt.format(class_name=name) for name in self.test_ds._label_names]
        else:
            return [self.text_prompt.format(class_name=name) for name in self.train_ds._label_names]