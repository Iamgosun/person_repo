from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Sequence

from torch.utils.data import Subset


def unwrap_dataset(ds: Any) -> Any:
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds



def unwrap_dataset_and_indices(ds: Any) -> tuple[Any, list[int] | None]:
    """
    把嵌套 Subset 展开成:
    - 最底层 base dataset
    - 对 base dataset 的最终索引映射
    """
    indices = None

    while isinstance(ds, Subset):
        cur = list(ds.indices)
        if indices is None:
            indices = cur
        else:
            indices = [cur[i] for i in indices]
        ds = ds.dataset

    return ds, indices


def get_class_names(ds: Any) -> list[str]:
    """
    尽量兼容项目里不同 dataset wrapper 的字段命名。
    """
    base_ds = unwrap_dataset(ds)

    class_names = getattr(base_ds, "classes", None)
    if class_names is None:
        class_names = getattr(base_ds, "_label_names", None)
    if class_names is None:
        class_names = getattr(base_ds, "label_names", None)
    if class_names is None:
        class_names = getattr(base_ds, "classnames", None)

    if class_names is None:
        raise ValueError("当前数据集无法自动提取类别名，请手动补充。")

    return list(class_names)


def extract_labels_fast(ds: Any) -> list[int]:
    """
    尽量避免通过 ds[i] 触发图片读取 / transform。
    只在拿不到底层标签结构时，才退回慢路径。
    """
    base_ds, indices = unwrap_dataset_and_indices(ds)

    # CIFAR10FolderDataset: self._samples = [(path, class_id), ...]
    if hasattr(base_ds, "_samples"):
        src = base_ds._samples
        idxs = indices if indices is not None else range(len(src))
        return [int(src[i][1]) for i in idxs]

    # SUN397 / UCF101: self._split_info = [(path, class_id, class_name), ...]
    if hasattr(base_ds, "_split_info"):
        src = base_ds._split_info
        idxs = indices if indices is not None else range(len(src))
        return [int(src[i][1]) for i in idxs]

    # Food101 / Flowers102 这类 torchvision 包装: _labels + indices
    if hasattr(base_ds, "_labels"):
        mapped = getattr(base_ds, "indices", None)
        if mapped is not None:
            idxs = indices if indices is not None else range(len(mapped))
            return [int(base_ds._labels[mapped[i]]) for i in idxs]
        idxs = indices if indices is not None else range(len(base_ds._labels))
        return [int(base_ds._labels[i]) for i in idxs]

    # CIFAR100 HF dataset wrapper
    if hasattr(base_ds, "_data"):
        idxs = indices if indices is not None else range(len(base_ds._data))
        labels = []
        for i in idxs:
            item = base_ds._data[i]
            if "fine_label" in item:
                labels.append(int(item["fine_label"]))
            elif "label" in item:
                labels.append(int(item["label"]))
            else:
                raise KeyError("HF dataset item 中找不到 fine_label / label 字段。")
        return labels

    # fallback: 走慢路径
    return [int(ds[i]["class_id"]) for i in range(len(ds))]


def build_fewshot_subset(
    ds: Any,
    shots_per_class: int,
    seed: int,
    strict: bool = True,
) -> Any:
    """
    从训练集按每类 K-shot 抽样。
    - shots_per_class <= 0: 直接返回原训练集
    - strict=True: 某类不足 K 个样本时直接报错
    """
    if shots_per_class <= 0:
        return ds

    labels = extract_labels_fast(ds)
    rng = random.Random(seed)

    indices = list(range(len(labels)))
    rng.shuffle(indices)

    picked = defaultdict(list)
    for idx in indices:
        label = int(labels[idx])
        if len(picked[label]) < shots_per_class:
            picked[label].append(idx)

    final_indices = []
    insufficient = {}
    all_labels = sorted(set(labels))

    for label in all_labels:
        selected = picked[label]
        if len(selected) < shots_per_class:
            insufficient[label] = len(selected)
        final_indices.extend(selected)

    if strict and insufficient:
        raise ValueError(f"few-shot 抽样失败，部分类别样本数不足：{insufficient}")

    final_indices = sorted(final_indices)
    return Subset(ds, final_indices)


def count_class_samples(ds: Any) -> tuple[list[str], Counter]:
    class_names = get_class_names(ds)
    labels = extract_labels_fast(ds)
    counter = Counter(labels)
    return class_names, counter


def print_class_counts(ds: Any, split_name: str = "train") -> tuple[list[str], Counter]:
    class_names, counter = count_class_samples(ds)

    print(f"===== {split_name} =====")
    print(f"num_classes: {len(class_names)}")
    for class_id in sorted(counter.keys()):
        print(f"{class_id:3d} | {class_names[class_id]:25s} | {counter[class_id]}")

    return class_names, counter


def check_dataset_schema(
    ds: Any,
    split_name: str,
    sample_count: int = 2,
    required_keys: Sequence[str] = ("image", "text", "class_id"),
) -> None:
    n = len(ds)
    print(f"[dataset] {split_name}: len={n}")

    if n == 0:
        raise ValueError(f"{split_name} 数据集为空。")

    inspect_count = min(sample_count, n)
    for i in range(inspect_count):
        sample = ds[i]
        if not isinstance(sample, dict):
            raise TypeError(f"{split_name}[{i}] 不是 dict，而是 {type(sample)}")

        missing = [k for k in required_keys if k not in sample]
        if missing:
            raise KeyError(f"{split_name}[{i}] 缺少字段：{missing}")

        image = sample["image"]
        text = sample["text"]
        class_id = sample["class_id"]

        print(
            f"[dataset-check] {split_name}[{i}] "
            f"image_type={type(image).__name__} "
            f"text_type={type(text).__name__} "
            f"class_id={class_id}"
        )