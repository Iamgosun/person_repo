import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader, Subset

from .common import default_collate_fn, default_transform


IMG_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"
}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def _sort_class_names(class_names):
    """
    若类别目录名全是数字字符串（如 1~102），按数值排序；
    否则按字符串排序。
    """
    if all(str(x).isdigit() for x in class_names):
        return sorted(class_names, key=lambda x: int(x))
    return sorted(class_names)


def _find_dataset_root(data_dir: Path) -> Path:
    """
    自动适配几种常见目录：
    1) data_dir/train, data_dir/valid, data_dir/test
    2) data_dir/flowers/train, ...
    3) data_dir/flowers/flowers/train, ...
    """
    candidates = [
        data_dir,
        data_dir / "flowers",
        data_dir / "flowers" / "flowers",
        data_dir / "flowers102",
    ]

    for root in candidates:
        train_root = root / "train"
        valid_root = root / "valid"
        test_root = root / "test"
        if train_root.is_dir() and valid_root.is_dir() and test_root.is_dir():
            return root

    raise FileNotFoundError(
        f"Cannot find Flowers102 split root under {data_dir}. "
        f"Expected directories like train/valid/test."
    )


def _find_cat_to_name_json(data_dir: Path, dataset_root: Path) -> Optional[Path]:
    """
    尝试寻找 cat_to_name.json：
    - dataset_root 下
    - data_dir 下
    - 父目录下
    - data_dir 递归搜索
    """
    candidates = [
        dataset_root / "cat_to_name.json",
        data_dir / "cat_to_name.json",
        dataset_root.parent / "cat_to_name.json",
        data_dir.parent / "cat_to_name.json" if data_dir.parent != data_dir else None,
    ]

    for p in candidates:
        if p is not None and p.exists():
            return p

    # 最后再做一次递归搜索
    for p in data_dir.rglob("cat_to_name.json"):
        if p.exists():
            return p

    return None


def _load_cat_to_name(json_path: Optional[Path]):
    if json_path is None:
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # 统一成 str -> str
    return {str(k): str(v) for k, v in mapping.items()}


def _find_split_class_names(split_root: Path):
    class_names = [p.name for p in split_root.iterdir() if p.is_dir()]
    if not class_names:
        raise FileNotFoundError(f"No class folders found under: {split_root}")
    return _sort_class_names(class_names)


def _build_class_to_idx(class_names):
    return {class_name: idx for idx, class_name in enumerate(class_names)}


def _check_split_classes(split_root: Path, expected_class_names):
    split_class_names = _find_split_class_names(split_root)
    if split_class_names != expected_class_names:
        raise ValueError(
            f"Class folders mismatch under {split_root}\n"
            f"expected={expected_class_names}\n"
            f"found={split_class_names}"
        )


def _build_samples_from_folder(split_root: Path, class_to_idx):
    samples = []

    for class_name in _sort_class_names(class_to_idx.keys()):
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        image_paths = sorted([p for p in class_dir.rglob("*") if _is_image_file(p)])
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No images found under class folder: {class_dir}")

        class_id = class_to_idx[class_name]
        samples.extend([(str(image_path), class_id) for image_path in image_paths])

    return samples


def _build_label_names(class_names, cat_to_name):
    """
    输出 label_names，顺序与 class_id 对齐。
    如果有 cat_to_name.json，则把 '1' -> 'pink primrose' 这种映射读进来；
    否则直接使用目录名本身。
    """
    label_names = []
    for class_name in class_names:
        if cat_to_name is not None and class_name in cat_to_name:
            label_names.append(cat_to_name[class_name])
        else:
            label_names.append(class_name)
    return label_names


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


class Flowers102LocalDataset(Dataset):
    def __init__(
        self,
        samples,
        label_names,
        text_prompt: str,
        transform=None,
    ):
        """
        samples: List[(image_path, class_id)]
        label_names: 类别名称列表，顺序与 class_id 对齐
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


class Flowers102DataModule(L.LightningDataModule):
    DATASET_SUBDIR = "flowers102"

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
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices

        self.use_few_shot = use_few_shot
        self.shots_per_class = shots_per_class
        self.few_shot_sample_seed = few_shot_sample_seed

        self.train_transform = train_transform or default_transform(image_size=224)
        self.test_transform = test_transform or default_transform(image_size=224)

        self.label_names = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str = None):
        dataset_root = _find_dataset_root(self.data_dir)

        train_root = dataset_root / "train"
        valid_root = dataset_root / "valid"
        test_root = dataset_root / "test"

        if not train_root.is_dir():
            raise FileNotFoundError(f"Flowers102 train root not found: {train_root}")
        if not valid_root.is_dir():
            raise FileNotFoundError(f"Flowers102 valid root not found: {valid_root}")
        if not test_root.is_dir():
            raise FileNotFoundError(f"Flowers102 test root not found: {test_root}")

        # 用 train 的类别目录作为全局类别定义
        class_names = _find_split_class_names(train_root)
        class_to_idx = _build_class_to_idx(class_names)

        # 检查 valid / test 的类别是否一致
        _check_split_classes(valid_root, class_names)
        _check_split_classes(test_root, class_names)

        # 尝试加载类别名映射
        cat_to_name_json = _find_cat_to_name_json(self.data_dir, dataset_root)
        cat_to_name = _load_cat_to_name(cat_to_name_json)

        self.label_names = _build_label_names(class_names, cat_to_name)

        train_samples = _build_samples_from_folder(train_root, class_to_idx)
        val_samples = _build_samples_from_folder(valid_root, class_to_idx)
        test_samples = _build_samples_from_folder(test_root, class_to_idx)

        train_ds = Flowers102LocalDataset(
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
        self.val_ds = Flowers102LocalDataset(
            val_samples,
            label_names=self.label_names,
            text_prompt=self.text_prompt,
            transform=self.test_transform,
        )
        self.test_ds = Flowers102LocalDataset(
            test_samples,
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