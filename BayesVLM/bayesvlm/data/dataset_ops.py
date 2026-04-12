from __future__ import annotations

import random
from collections import Counter, defaultdict
from copy import copy
from typing import Any, Sequence

from torch.utils.data import Subset


def unwrap_dataset(ds: Any) -> Any:
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def unwrap_dataset_and_indices(ds: Any) -> tuple[Any, list[int] | None]:
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
    if hasattr(ds, "class_names"):
        return list(ds.class_names)
    base_ds = unwrap_dataset(ds)
    for attr in ["classes", "_label_names", "label_names", "classnames", "class_names"]:
        class_names = getattr(base_ds, attr, None)
        if class_names is not None:
            return list(class_names)
    raise ValueError("could not infer class names from dataset")


def extract_labels_fast(ds: Any) -> list[int]:
    if hasattr(ds, "labels"):
        return [int(x) for x in ds.labels]
    base_ds, indices = unwrap_dataset_and_indices(ds)
    if hasattr(base_ds, "_samples"):
        src = base_ds._samples
        idxs = indices if indices is not None else range(len(src))
        return [int(src[i][1]) for i in idxs]
    if hasattr(base_ds, "_split_info"):
        src = base_ds._split_info
        idxs = indices if indices is not None else range(len(src))
        return [int(src[i][1]) for i in idxs]
    if hasattr(base_ds, "_labels"):
        mapped = getattr(base_ds, "indices", None)
        if mapped is not None:
            idxs = indices if indices is not None else range(len(mapped))
            return [int(base_ds._labels[mapped[i]]) for i in idxs]
        idxs = indices if indices is not None else range(len(base_ds._labels))
        return [int(base_ds._labels[i]) for i in idxs]
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
                raise KeyError("HF dataset item missing fine_label / label")
        return labels
    return [int(ds[i]["class_id"]) for i in range(len(ds))]


def build_fewshot_subset(ds: Any, shots_per_class: int, seed: int, strict: bool = True) -> Any:
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
        raise ValueError(f"few-shot sampling failed, insufficient class counts: {insufficient}")
    return Subset(ds, sorted(final_indices))


def count_class_samples(ds: Any) -> tuple[list[str], Counter]:
    class_names = get_class_names(ds)
    labels = extract_labels_fast(ds)
    return class_names, Counter(labels)


def print_class_counts(ds: Any, split_name: str = "train") -> tuple[list[str], Counter]:
    class_names, counter = count_class_samples(ds)
    print(f"===== {split_name} =====")
    print(f"num_classes: {len(class_names)}")
    for class_id in sorted(counter.keys()):
        print(f"{class_id:3d} | {class_names[class_id]:25s} | {counter[class_id]}")
    return class_names, counter


def clone_sample_dict(sample: dict[str, Any]) -> dict[str, Any]:
    new_sample = copy(sample)
    return new_sample
