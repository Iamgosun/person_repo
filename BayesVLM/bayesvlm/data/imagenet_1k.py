from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytorch_lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from .common import default_collate_fn, default_transform


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


def _find_meta_mat(dataset_root: Path) -> Optional[Path]:
    candidates = [
        dataset_root / "devkit" / "ILSVRC2012_devkit_t12" / "data" / "meta.mat",
        dataset_root / "devkit" / "data" / "meta.mat",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_imagenet_readable_names(dataset_root: Path, wnids):
    """
    尝试从 devkit/meta.mat 读取可读类别名。
    读不到时退化为 wnid 本身。
    """
    wnid_to_name = {wnid: wnid for wnid in wnids}

    meta_path = _find_meta_mat(dataset_root)
    if meta_path is None:
        return [wnid_to_name[wnid] for wnid in wnids]

    try:
        import scipy.io

        meta = scipy.io.loadmat(meta_path, squeeze_me=True)["synsets"]
        for entry in meta:
            # 常见结构:
            # entry[1] -> wnid
            # entry[2] -> words
            # entry[4] -> num_children
            wnid = str(entry[1])
            words = str(entry[2])
            num_children = int(entry[4])

            # 只保留 1000 个叶子节点类别
            if num_children == 0 and wnid in wnid_to_name:
                wnid_to_name[wnid] = words.split(",")[0].strip()
    except Exception:
        pass

    return [wnid_to_name[wnid] for wnid in wnids]


def _resolve_dataset_root(data_dir: Path, subdir_name: str) -> Path:
    """
    兼容两种传法：
    1) data_dir=/.../imagenet
    2) data_dir=/.../datasets   且下面有 imagenet/
    """
    if (data_dir / "train").is_dir():
        return data_dir

    candidate = data_dir / subdir_name
    if (candidate / "train").is_dir():
        return candidate

    return data_dir


def _select_subset_class_ids(
    all_wnids,
    class_to_idx,
    num_classes: Optional[int],
    class_seed: int,
    class_wids: Optional[Sequence[str]],
):
    if class_wids is not None:
        missing = [wnid for wnid in class_wids if wnid not in class_to_idx]
        if missing:
            raise ValueError(f"Unknown class_wids: {missing[:10]}")
        selected_class_ids = sorted(class_to_idx[wnid] for wnid in class_wids)
        return selected_class_ids

    total_classes = len(all_wnids)
    if num_classes is None or num_classes >= total_classes:
        return list(range(total_classes))

    rng = np.random.default_rng(class_seed)
    selected_class_ids = rng.choice(total_classes, size=num_classes, replace=False)
    selected_class_ids = np.sort(selected_class_ids).tolist()
    return selected_class_ids


def _filter_samples_by_class_ids(samples, allowed_class_ids):
    allowed = set(allowed_class_ids)
    return [(path, class_id) for path, class_id in samples if class_id in allowed]


class ImageNetFolderDataset(Dataset):
    def __init__(
        self,
        samples,
        text_prompt: str,
        class_id_to_class_name: dict,
        class_id_to_subset_class_id: dict,
        class_id_to_wnid: dict,
        transform=None,
    ):
        self._samples = samples
        self._text_prompt = text_prompt
        self._transform = transform
        self._class_id_to_class_name = class_id_to_class_name
        self._class_id_to_subset_class_id = class_id_to_subset_class_id
        self._class_id_to_wnid = class_id_to_wnid
        self._label_names = list(self._class_id_to_class_name.values())

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        image_path, original_class_id = self._samples[idx]

        image = Image.open(image_path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)

        class_name = self._class_id_to_class_name[original_class_id]
        subset_class_id = self._class_id_to_subset_class_id[original_class_id]
        wnid = self._class_id_to_wnid[original_class_id]

        text = self._text_prompt.format(class_name=class_name)

        return {
            "image": image,
            "text": text,
            "class_id": subset_class_id,
            "class_name": class_name,
            "class_wnid": wnid,
            "image_id": idx,
            "image_path": str(image_path),
        }


class Imagenet1kDataModule(L.LightningDataModule):
    DATASET_SUBDIR = "imagenet"

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
        class_seed: int = 42,
        num_classes: Optional[int] = 1000,
        class_wids: Sequence[str] = None,
        shots_per_class: int = 10,
        use_few_shot: bool = False,
        few_shot_sample_seed: int = 42,
        val_split: float = 0.2,
        split_seed: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices

        self.class_seed = class_seed
        self.num_classes = num_classes
        self.class_wids = class_wids

        self.shots_per_class = shots_per_class
        self.use_few_shot = use_few_shot
        self.few_shot_sample_seed = few_shot_sample_seed

        self.val_split = val_split
        self.split_seed = split_seed

        self.label_names = None
        self.label_wnids = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        if self.class_wids is not None:
            print("Using custom class WIDs: this will override the num_classes parameter")

    def setup(self, stage: str = None):
        dataset_root = _resolve_dataset_root(self.data_dir, self.DATASET_SUBDIR)

        train_root = dataset_root / "train"
        test_root = dataset_root / "val"   # 官方 val 当 test

        if not train_root.is_dir():
            raise FileNotFoundError(f"ImageNet train root not found: {train_root}")
        if not test_root.is_dir():
            raise FileNotFoundError(f"ImageNet val root not found: {test_root}")

        train_folder = ImageFolder(train_root)

        all_wnids = train_folder.classes
        all_label_names = _load_imagenet_readable_names(dataset_root, all_wnids)
        class_to_idx = train_folder.class_to_idx

        selected_class_ids = _select_subset_class_ids(
            all_wnids=all_wnids,
            class_to_idx=class_to_idx,
            num_classes=self.num_classes,
            class_seed=self.class_seed,
            class_wids=self.class_wids,
        )

        self.label_wnids = [all_wnids[i] for i in selected_class_ids]
        self.label_names = [all_label_names[i] for i in selected_class_ids]

        class_id_to_class_name = OrderedDict(
            (orig_class_id, all_label_names[orig_class_id]) for orig_class_id in selected_class_ids
        )
        class_id_to_subset_class_id = OrderedDict(
            (orig_class_id, subset_class_id)
            for subset_class_id, orig_class_id in enumerate(selected_class_ids)
        )
        class_id_to_wnid = OrderedDict(
            (orig_class_id, all_wnids[orig_class_id]) for orig_class_id in selected_class_ids
        )

        all_train_samples = _filter_samples_by_class_ids(
            train_folder.samples,
            selected_class_ids,
        )

        n_total = len(all_train_samples)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(self.split_seed)
        perm = torch.randperm(n_total, generator=generator).tolist()

        train_indices = perm[:n_train]
        val_indices = perm[n_train:]

        train_samples = [all_train_samples[i] for i in train_indices]
        val_samples = [all_train_samples[i] for i in val_indices]

        train_ds = ImageNetFolderDataset(
            train_samples,
            text_prompt=self.text_prompt,
            class_id_to_class_name=class_id_to_class_name,
            class_id_to_subset_class_id=class_id_to_subset_class_id,
            class_id_to_wnid=class_id_to_wnid,
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

        self.val_ds = ImageNetFolderDataset(
            val_samples,
            text_prompt=self.text_prompt,
            class_id_to_class_name=class_id_to_class_name,
            class_id_to_subset_class_id=class_id_to_subset_class_id,
            class_id_to_wnid=class_id_to_wnid,
            transform=self.test_transform,
        )

        test_folder = ImageFolder(test_root)
        if test_folder.classes != all_wnids:
            raise ValueError(
                f"Train/Test class names mismatch:\n"
                f"train={all_wnids[:5]} ...\n"
                f"test={test_folder.classes[:5]} ..."
            )

        test_samples = _filter_samples_by_class_ids(
            test_folder.samples,
            selected_class_ids,
        )

        self.test_ds = ImageNetFolderDataset(
            test_samples,
            text_prompt=self.text_prompt,
            class_id_to_class_name=class_id_to_class_name,
            class_id_to_subset_class_id=class_id_to_subset_class_id,
            class_id_to_wnid=class_id_to_wnid,
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


class Imagenet100DataModule(Imagenet1kDataModule):
    def __init__(self, **kwargs):
        super().__init__(num_classes=100, **kwargs)


class Imagenet50DataModule(Imagenet1kDataModule):
    def __init__(self, **kwargs):
        super().__init__(num_classes=50, **kwargs)