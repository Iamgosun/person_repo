from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from bayesvlm.data.common import default_collate_fn
from bayesvlm.data.dataset_ops import (
    build_fewshot_subset,
    check_dataset_schema,
    get_class_names,
)
from bayesvlm.data.factory import DataModuleFactory


@dataclass
class ExperimentDataBundle:
    datamodule: Any
    raw_train_ds: Any
    train_ds: Any
    val_ds: Any
    test_ds: Any
    class_names: list[str]
    train_loader: DataLoader
    train_eval_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def build_loader(
    ds: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool | None = None,
    drop_last: bool = False,
):
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


def probe_loader(loader: DataLoader, split_name: str) -> None:
    batch = next(iter(loader))

    keys = list(batch.keys())
    print(f"[loader-probe] {split_name}: keys={keys}")

    if "image" in batch and torch.is_tensor(batch["image"]):
        print(f"[loader-probe] {split_name}: image_shape={tuple(batch['image'].shape)}")
    if "class_id" in batch and torch.is_tensor(batch["class_id"]):
        print(f"[loader-probe] {split_name}: class_id_shape={tuple(batch['class_id'].shape)}")
    if "text" in batch and isinstance(batch["text"], (list, tuple)) and len(batch["text"]) > 0:
        print(f"[loader-probe] {split_name}: text_example={batch['text'][0]!r}")
    if "image_path" in batch and isinstance(batch["image_path"], (list, tuple)) and len(batch["image_path"]) > 0:
        print(f"[loader-probe] {split_name}: image_path_example={batch['image_path'][0]!r}")


def create_datamodule(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    train_transform: Any,
    test_transform: Any,
    shuffle_train: bool = True,
) -> Any:
    factory = DataModuleFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
        test_transform=test_transform,
        shuffle_train=shuffle_train,
        base_path=data_root,
    )
    return factory.create(dataset)


def prepare_experiment_data(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    train_transform: Any,
    test_transform: Any,
    shots_per_class: int,
    seed: int,
    shuffle_train: bool = True,
    run_checks: bool = False,
    run_loader_probe: bool = False,
) -> ExperimentDataBundle:
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

    raw_train_ds = dm.train_ds
    val_ds = dm.val_ds if hasattr(dm, "val_ds") and dm.val_ds is not None else dm.test_ds
    test_ds = dm.test_ds

    train_ds = build_fewshot_subset(
        raw_train_ds,
        shots_per_class=shots_per_class,
        seed=seed,
        strict=True,
    )

    class_names = get_class_names(train_ds)

    if run_checks:
        check_dataset_schema(train_ds, "train")
        check_dataset_schema(val_ds, "val")
        check_dataset_schema(test_ds, "test")

    train_loader = build_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle_train,
    )
    train_eval_loader = build_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    val_loader = build_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    test_loader = build_loader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    if run_loader_probe:
        probe_loader(train_loader, "train")
        probe_loader(val_loader, "val")
        probe_loader(test_loader, "test")

    return ExperimentDataBundle(
        datamodule=dm,
        raw_train_ds=raw_train_ds,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        class_names=class_names,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )