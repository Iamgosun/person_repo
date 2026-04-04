import os
from typing import Sequence, Optional
from PIL import Image

import torch
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder

from .common import default_collate_fn, default_transform


class CIFAR10FolderDataset(Dataset):
    def __init__(
        self,
        samples,
        label_names,
        text_prompt: str,
        transform=None,
    ):
        """
        samples: List[(image_path, class_id)]
        label_names: 类别名列表，如 ['airplane', 'automobile', ...]
        """
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

        text = self._text_prompt.format(
            class_name=self._label_names[class_id]
        )

        return {
            "image": image,
            "text": text,
            "class_id": class_id,
            "image_id": idx,
            "image_path": image_path,
        }
    



class CIFAR10DataModule(L.LightningDataModule):
    DATASET_SUBDIR = "cifar10"

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
        val_split: float = 0.2,
        split_seed: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.text_prompt = text_prompt
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices
        self.val_split = val_split
        self.split_seed = split_seed

        self.train_transform = train_transform or default_transform(image_size=224)
        self.test_transform = test_transform or default_transform(image_size=224)

        self.label_names = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str = None):
        dataset_root = self.data_dir
        #dataset_root = os.path.join(self.data_dir, self.DATASET_SUBDIR)
        train_root = os.path.join(dataset_root, "train")
        test_root = os.path.join(dataset_root, "test")

        # 读取 train 文件夹，拿到类别和所有样本
        train_folder = ImageFolder(train_root)
        self.label_names = train_folder.classes
        all_train_samples = train_folder.samples  # [(path, class_id), ...]

        # 划分 train / val
        n_total = len(all_train_samples)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(self.split_seed)
        perm = torch.randperm(n_total, generator=generator).tolist()

        train_indices = perm[:n_train]
        val_indices = perm[n_train:]

        train_samples = [all_train_samples[i] for i in train_indices]
        val_samples = [all_train_samples[i] for i in val_indices]

        self.train_ds = CIFAR10FolderDataset(
            train_samples,
            label_names=self.label_names,
            text_prompt=self.text_prompt,
            transform=self.train_transform,
        )

        if self.subset_indices is not None:
            self.train_ds = Subset(self.train_ds, self.subset_indices)

        self.val_ds = CIFAR10FolderDataset(
            val_samples,
            label_names=self.label_names,
            text_prompt=self.text_prompt,
            transform=self.test_transform,
        )

        # test 集
        test_folder = ImageFolder(test_root)

        # 最好检查 train/test 的类别顺序是否一致
        if test_folder.classes != self.label_names:
            raise ValueError(
                f"Train/Test class names mismatch:\n"
                f"train={self.label_names}\n"
                f"test={test_folder.classes}"
            )

        self.test_ds = CIFAR10FolderDataset(
            test_folder.samples,
            label_names=self.label_names,
            text_prompt=self.text_prompt,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=default_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=default_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=default_collate_fn,
        )

    @property
    def class_prompts(self):
        if self.label_names is None:
            raise RuntimeError("Call setup() before accessing class_prompts.")
        return [self.text_prompt.format(class_name=name) for name in self.label_names]