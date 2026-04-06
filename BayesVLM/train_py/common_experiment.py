from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesvlm.data.dataset_ops import print_class_counts, unwrap_dataset_and_indices
from bayesvlm.data.pipeline import build_loader, prepare_experiment_data
from bayesvlm.features.feature_dataset import build_feature_loader
from bayesvlm.features.image_cache import (
    ImageFeatureCacheSpec,
    get_or_build_image_feature_bundle,
)
from bayesvlm.training.io import save_json
from bayesvlm.utils import (
    get_image_size,
    get_model_type_and_size,
    get_transform,
    load_model,
)


@dataclass
class ExperimentContext:
    run_dir: Path
    data: Any
    class_names: list[str]
    image_encoder: Any
    text_encoder: Any
    vlm: Any
    train_loader: Any
    train_eval_loader: Any
    val_loader: Any
    test_loader: Any
    model_type: str
    transform_image_size: int
    transform_name: str
    local_model_path_cache_key: str | None
    data_root_path: Path
    image_feature_cache_root_path: Path
    common_config: dict[str, Any]


def resolve_existing_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None

    p = Path(path_str)
    if p.exists():
        return p.resolve()

    p2 = (Path.cwd() / path_str).resolve()
    if p2.exists():
        return p2

    raise FileNotFoundError(f"路径不存在：{path_str}")


def normalize_path_for_cache(path_str: str | None) -> str | None:
    """
    用于 cache key 的稳定路径：
    - 真正存在的本地路径 -> 绝对路径
    - 不存在的字符串（例如未来可能的 repo id）-> 原样保留
    """
    if path_str is None:
        return None

    p = Path(path_str)
    if p.exists():
        return str(p.resolve())

    p2 = (Path.cwd() / path_str).resolve()
    if p2.exists():
        return str(p2)

    return path_str


def stable_transform_name(model_type: str, image_size: int) -> str:
    """
    不使用 repr(transform)，避免函数对象地址进入 cache key。
    """
    if model_type == "siglip":
        return f"siglip_transform(image_size={image_size})"
    return f"default_transform(image_size={image_size})"


def build_common_context(
    *,
    args,
    run_dir: Path,
    require_image_feature_cache: bool = False,
) -> ExperimentContext:
    model_type, _ = get_model_type_and_size(args.model)
    transform_image_size = get_image_size(args.model)
    transform = get_transform(model_type, transform_image_size)
    transform_name = stable_transform_name(model_type, transform_image_size)

    data_root_path = resolve_existing_path(args.data_root)
    local_model_path_cache_key = normalize_path_for_cache(args.local_model_path)
    image_feature_cache_root_path = Path(args.image_feature_cache_root).resolve()

    data = prepare_experiment_data(
        dataset=args.dataset,
        data_root=str(data_root_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=transform,
        test_transform=transform,
        shots_per_class=args.shots_per_class,
        seed=args.seed,
        shuffle_train=True,
        run_checks=False,
        run_loader_probe=False,
    )

    print_class_counts(data.train_ds, split_name="train")
    print_class_counts(data.test_ds, split_name="test")
    save_json(run_dir / "class_names.json", {"class_names": data.class_names})

    image_encoder, text_encoder, vlm = load_model(
        model_str=args.model,
        device=args.device,
        local_model_path=args.local_model_path,
    )

    if not args.cache_image_features and require_image_feature_cache:
        raise ValueError(
            "当前 recipe 依赖缓存图像特征。"
            "请不要关闭 cache_image_features，或者为该方法单独实现 raw-image 训练路径。"
        )

    if args.cache_image_features:
        print("[0] 构建/加载共享图像特征缓存 ...")

        raw_train_loader_for_cache = build_loader(
            data.raw_train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        train_full_features = get_or_build_image_feature_bundle(
            image_encoder=image_encoder,
            loader=raw_train_loader_for_cache,
            ds=data.raw_train_ds,
            cache_root=image_feature_cache_root_path,
            spec=ImageFeatureCacheSpec(
                dataset=args.dataset,
                split="train_full",
                model_str=args.model,
                local_model_path=local_model_path_cache_key,
                image_size=transform_image_size,
                transform_name=transform_name,
                data_root=str(data_root_path),
            ),
            force_rebuild=args.rebuild_image_feature_cache,
        )

        val_features = get_or_build_image_feature_bundle(
            image_encoder=image_encoder,
            loader=data.val_loader,
            ds=data.val_ds,
            cache_root=image_feature_cache_root_path,
            spec=ImageFeatureCacheSpec(
                dataset=args.dataset,
                split="val",
                model_str=args.model,
                local_model_path=local_model_path_cache_key,
                image_size=transform_image_size,
                transform_name=transform_name,
                data_root=str(data_root_path),
            ),
            force_rebuild=args.rebuild_image_feature_cache,
        )

        test_features = get_or_build_image_feature_bundle(
            image_encoder=image_encoder,
            loader=data.test_loader,
            ds=data.test_ds,
            cache_root=image_feature_cache_root_path,
            spec=ImageFeatureCacheSpec(
                dataset=args.dataset,
                split="test",
                model_str=args.model,
                local_model_path=local_model_path_cache_key,
                image_size=transform_image_size,
                transform_name=transform_name,
                data_root=str(data_root_path),
            ),
            force_rebuild=args.rebuild_image_feature_cache,
        )

        _, train_indices = unwrap_dataset_and_indices(data.train_ds)
        if train_indices is None:
            train_indices = list(range(len(data.train_ds)))

        print(
            f"[cache] raw_train_samples={len(data.raw_train_ds)} | "
            f"fewshot_train_samples={len(train_indices)}"
        )

        train_subset_features = train_full_features.subset(train_indices)

        train_loader = build_feature_loader(
            train_subset_features,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        train_eval_loader = build_feature_loader(
            train_subset_features,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        val_loader = build_feature_loader(
            val_features,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        test_loader = build_feature_loader(
            test_features,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    else:
        train_loader = data.train_loader
        train_eval_loader = data.train_eval_loader
        val_loader = data.val_loader
        test_loader = data.test_loader

    common_config = {
        "dataset": args.dataset,
        "model_str": args.model,
        "local_model_path_raw": args.local_model_path,
        "local_model_path_cache_key": local_model_path_cache_key,
        "data_root_raw": args.data_root,
        "data_root_cache_key": str(data_root_path),
        "model_type": model_type,
        "image_size": transform_image_size,
        "transform_name_cache_key": transform_name,
        "shots_per_class": args.shots_per_class,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "prediction_topk": args.prediction_topk,
        "cache_image_features": args.cache_image_features,
        "image_feature_cache_root": str(image_feature_cache_root_path),
        "rebuild_image_feature_cache": args.rebuild_image_feature_cache,
        "num_classes": len(data.class_names),
        "run_dir": str(run_dir),
    }

    return ExperimentContext(
        run_dir=run_dir,
        data=data,
        class_names=data.class_names,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        vlm=vlm,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_type=model_type,
        transform_image_size=transform_image_size,
        transform_name=transform_name,
        local_model_path_cache_key=local_model_path_cache_key,
        data_root_path=data_root_path,
        image_feature_cache_root_path=image_feature_cache_root_path,
        common_config=common_config,
    )