from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bayesvlm.data.dataset_ops import extract_labels_fast, get_class_names, clone_sample_dict


class RelabeledSubsetDataset:
    def __init__(self, base_ds: Any, indices: list[int], relabeler: dict[int, int], class_names: list[str]):
        self.base_ds = base_ds
        self.indices = list(indices)
        self.relabeler = {int(k): int(v) for k, v in relabeler.items()}
        self.class_names = list(class_names)
        self.labels = [self.relabeler[int(extract_labels_fast(base_ds)[i])] for i in self.indices]
        self.sample_keys = None
        self.image_paths = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        sample = self.base_ds[base_idx]
        label_old = int(sample["class_id"])
        sample = clone_sample_dict(sample)
        sample["orig_class_id"] = label_old
        sample["class_id"] = self.relabeler[label_old]
        sample["class_name"] = self.class_names[sample["class_id"]]
        return sample


@dataclass
class Base2NewSplit:
    base_train_full: Any
    base_val: Any
    base_test: Any
    new_test: Any
    base_class_names: list[str]
    new_class_names: list[str]
    base_labels: list[int]
    new_labels: list[int]


def split_base_new_labels(num_classes: int) -> tuple[list[int], list[int]]:
    if num_classes < 2:
        raise ValueError("base2new requires at least 2 classes")
    labels = list(range(num_classes))
    midpoint = (num_classes + 1) // 2
    return labels[:midpoint], labels[midpoint:]


def subset_and_relabel_dataset(ds: Any, selected_labels: list[int]) -> tuple[Any, list[str], dict[int, int]]:
    all_labels = extract_labels_fast(ds)
    all_class_names = get_class_names(ds)
    selected = [int(x) for x in selected_labels]
    relabeler = {label: new_id for new_id, label in enumerate(selected)}
    selected_set = set(selected)
    indices = [idx for idx, label in enumerate(all_labels) if int(label) in selected_set]
    class_names = [all_class_names[label] for label in selected]
    return RelabeledSubsetDataset(ds, indices=indices, relabeler=relabeler, class_names=class_names), class_names, relabeler


def build_base2new_split(train_ds: Any, val_ds: Any, test_ds: Any) -> Base2NewSplit:
    num_classes = len(get_class_names(train_ds))
    base_labels, new_labels = split_base_new_labels(num_classes)
    base_train_full, base_class_names, _ = subset_and_relabel_dataset(train_ds, base_labels)
    base_val, _, _ = subset_and_relabel_dataset(val_ds, base_labels)
    base_test, _, _ = subset_and_relabel_dataset(test_ds, base_labels)
    new_test, new_class_names, _ = subset_and_relabel_dataset(test_ds, new_labels)
    return Base2NewSplit(
        base_train_full=base_train_full,
        base_val=base_val,
        base_test=base_test,
        new_test=new_test,
        base_class_names=base_class_names,
        new_class_names=new_class_names,
        base_labels=base_labels,
        new_labels=new_labels,
    )
