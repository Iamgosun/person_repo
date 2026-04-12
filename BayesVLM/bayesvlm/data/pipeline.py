from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from bayesvlm.data.common import default_collate_fn
from bayesvlm.data.dataset_ops import build_fewshot_subset, get_class_names
from bayesvlm.data.factory import DataModuleFactory


@dataclass
class RawDataBundle:
    datamodule: Any
    train_ds: Any
    val_ds: Any
    test_ds: Any
    class_names: list[str]


@dataclass
class PreparedDataBundle:
    dataset_name: str
    train_ds: Any
    train_eval_ds: Any
    val_ds: Any
    test_ds: Any
    class_names: list[str]
    extra_eval_datasets: dict[str, tuple[Any, list[str]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentDataBundle:
    raw: RawDataBundle
    prepared: PreparedDataBundle
    train_loader: DataLoader
    train_eval_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def build_loader(ds: Any, batch_size: int, num_workers: int, shuffle: bool, pin_memory: bool | None = None, drop_last: bool = False):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
        collate_fn=default_collate_fn,
    )


def create_datamodule(dataset: str, data_root: str, batch_size: int, num_workers: int, train_transform: Any, test_transform: Any, shuffle_train: bool = True) -> Any:
    factory = DataModuleFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
        test_transform=test_transform,
        shuffle_train=shuffle_train,
        base_path=data_root,
    )
    return factory.create(dataset)


def prepare_raw_data_bundle(dataset: str, data_root: str, batch_size: int, num_workers: int, train_transform: Any, test_transform: Any, shuffle_train: bool = True) -> RawDataBundle:
    dm = create_datamodule(
        dataset=dataset,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
        test_transform=test_transform,
        shuffle_train=shuffle_train,
    )
    dm.setup()
    train_ds = dm.train_ds
    val_ds = dm.val_ds if hasattr(dm, "val_ds") and dm.val_ds is not None else dm.test_ds
    test_ds = dm.test_ds
    class_names = get_class_names(train_ds)
    return RawDataBundle(datamodule=dm, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, class_names=class_names)


def build_id_prepared_bundle(raw: RawDataBundle, dataset_name: str, shots_per_class: int, seed: int) -> PreparedDataBundle:
    train_ds = build_fewshot_subset(raw.train_ds, shots_per_class=shots_per_class, seed=seed, strict=True)
    class_names = get_class_names(train_ds)
    return PreparedDataBundle(
        dataset_name=dataset_name,
        train_ds=train_ds,
        train_eval_ds=train_ds,
        val_ds=raw.val_ds,
        test_ds=raw.test_ds,
        class_names=class_names,
        metadata={
            "train_split_name": "train",
            "val_split_name": "val",
            "test_split_name": "test",
            "protocol_train_tag": "id_train",
            "protocol_val_tag": "id_val",
            "protocol_test_tag": "id_test",
        },
    )
