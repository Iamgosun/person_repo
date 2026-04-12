from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesvlm.data.dataset_ops import print_class_counts
from bayesvlm.data.pipeline import PreparedDataBundle, build_loader
from bayesvlm.features.feature_dataset import build_feature_loader
from bayesvlm.features.image_cache import ImageFeatureCacheSpec, get_or_build_image_feature_bundle
from bayesvlm.training.io import save_json
from bayesvlm.utils import get_image_size, get_model_type_and_size, load_model


@dataclass
class ExperimentContext:
    run_dir: Path
    prepared: PreparedDataBundle
    class_names: list[str]
    image_encoder: Any
    text_encoder: Any
    vlm: Any
    train_loader: Any
    train_eval_loader: Any
    val_loader: Any
    test_loader: Any
    extra_eval_loaders: dict[str, Any]
    extra_eval_class_names: dict[str, list[str]]
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
    raise FileNotFoundError(f"path does not exist: {path_str}")


def normalize_path_for_cache(path_str: str | None) -> str | None:
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
    if model_type == "siglip":
        return f"siglip_transform(image_size={image_size})"
    return f"default_transform(image_size={image_size})"


def _build_feature_or_raw_loader(*, ds, dataset_name: str, split_tag: str, image_encoder, batch_size: int, num_workers: int, cache_image_features: bool, image_feature_cache_root_path: Path, force_rebuild: bool, model_str: str, local_model_path_cache_key: str | None, image_size: int, transform_name: str, data_root: str, shuffle: bool):
    raw_loader = build_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    if not cache_image_features:
        return build_loader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    bundle = get_or_build_image_feature_bundle(
        image_encoder=image_encoder,
        loader=raw_loader,
        ds=ds,
        cache_root=image_feature_cache_root_path,
        spec=ImageFeatureCacheSpec(
            dataset=dataset_name,
            split=split_tag,
            model_str=model_str,
            local_model_path=local_model_path_cache_key,
            image_size=image_size,
            transform_name=transform_name,
            data_root=data_root,
        ),
        force_rebuild=force_rebuild,
    )
    return build_feature_loader(bundle, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def build_common_context(*, args, run_dir: Path, prepared: PreparedDataBundle, require_image_feature_cache: bool = False) -> ExperimentContext:
    model_type, _ = get_model_type_and_size(args.model)
    transform_image_size = get_image_size(args.model)
    transform_name = stable_transform_name(model_type, transform_image_size)

    data_root_path = resolve_existing_path(args.data_root)
    local_model_path_cache_key = normalize_path_for_cache(args.local_model_path)
    image_feature_cache_root_path = Path(args.image_feature_cache_root).resolve()

    if not args.cache_image_features and require_image_feature_cache:
        raise ValueError("this family requires cached image features; cache_image_features must remain enabled")

    print_class_counts(prepared.train_ds, split_name="train")
    print_class_counts(prepared.test_ds, split_name="test")

    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    save_json(config_dir / "class_names.json", {"class_names": prepared.class_names})
    if prepared.extra_eval_datasets:
        save_json(config_dir / "extra_eval_class_names.json", {k: v[1] for k, v in prepared.extra_eval_datasets.items()})

    image_encoder, text_encoder, vlm = load_model(
        model_str=args.model,
        device=args.device,
        local_model_path=args.local_model_path,
    )

    train_loader = _build_feature_or_raw_loader(
        ds=prepared.train_ds,
        dataset_name=prepared.dataset_name,
        split_tag=f"{args.protocol}_train",
        image_encoder=image_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_image_features=args.cache_image_features,
        image_feature_cache_root_path=image_feature_cache_root_path,
        force_rebuild=args.rebuild_image_feature_cache,
        model_str=args.model,
        local_model_path_cache_key=local_model_path_cache_key,
        image_size=transform_image_size,
        transform_name=transform_name,
        data_root=str(data_root_path),
        shuffle=True,
    )
    train_eval_loader = _build_feature_or_raw_loader(
        ds=prepared.train_eval_ds,
        dataset_name=prepared.dataset_name,
        split_tag=f"{args.protocol}_train_eval",
        image_encoder=image_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_image_features=args.cache_image_features,
        image_feature_cache_root_path=image_feature_cache_root_path,
        force_rebuild=args.rebuild_image_feature_cache,
        model_str=args.model,
        local_model_path_cache_key=local_model_path_cache_key,
        image_size=transform_image_size,
        transform_name=transform_name,
        data_root=str(data_root_path),
        shuffle=False,
    )
    val_loader = _build_feature_or_raw_loader(
        ds=prepared.val_ds,
        dataset_name=prepared.dataset_name,
        split_tag=f"{args.protocol}_val",
        image_encoder=image_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_image_features=args.cache_image_features,
        image_feature_cache_root_path=image_feature_cache_root_path,
        force_rebuild=args.rebuild_image_feature_cache,
        model_str=args.model,
        local_model_path_cache_key=local_model_path_cache_key,
        image_size=transform_image_size,
        transform_name=transform_name,
        data_root=str(data_root_path),
        shuffle=False,
    )
    test_loader = _build_feature_or_raw_loader(
        ds=prepared.test_ds,
        dataset_name=prepared.dataset_name,
        split_tag=f"{args.protocol}_test",
        image_encoder=image_encoder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_image_features=args.cache_image_features,
        image_feature_cache_root_path=image_feature_cache_root_path,
        force_rebuild=args.rebuild_image_feature_cache,
        model_str=args.model,
        local_model_path_cache_key=local_model_path_cache_key,
        image_size=transform_image_size,
        transform_name=transform_name,
        data_root=str(data_root_path),
        shuffle=False,
    )

    extra_eval_loaders = {}
    extra_eval_class_names = {}
    for name, (ds, class_names) in prepared.extra_eval_datasets.items():
        extra_eval_loaders[name] = _build_feature_or_raw_loader(
            ds=ds,
            dataset_name=name,
            split_tag=f"{args.protocol}_{name}",
            image_encoder=image_encoder,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_image_features=args.cache_image_features,
            image_feature_cache_root_path=image_feature_cache_root_path,
            force_rebuild=args.rebuild_image_feature_cache,
            model_str=args.model,
            local_model_path_cache_key=local_model_path_cache_key,
            image_size=transform_image_size,
            transform_name=transform_name,
            data_root=str(data_root_path),
            shuffle=False,
        )
        extra_eval_class_names[name] = list(class_names)

    common_config = {
        "dataset": prepared.dataset_name,
        "model": args.model,
        "local_model_path": args.local_model_path,
        "data_root": args.data_root,
        "model_type": model_type,
        "image_size": transform_image_size,
        "transform_name": transform_name,
        "shots_per_class": args.shots_per_class,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "prediction_topk": args.prediction_topk,
        "cache_image_features": args.cache_image_features,
        "image_feature_cache_root": str(image_feature_cache_root_path),
        "rebuild_image_feature_cache": args.rebuild_image_feature_cache,
        "num_classes": len(prepared.class_names),
        "run_dir": str(run_dir),
        "family": args.family,
        "variant": args.variant,
        "protocol": args.protocol,
        "evaluation_tasks": list(args.evaluation_tasks),
    }

    return ExperimentContext(
        run_dir=run_dir,
        prepared=prepared,
        class_names=prepared.class_names,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        vlm=vlm,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        extra_eval_loaders=extra_eval_loaders,
        extra_eval_class_names=extra_eval_class_names,
        model_type=model_type,
        transform_image_size=transform_image_size,
        transform_name=transform_name,
        local_model_path_cache_key=local_model_path_cache_key,
        data_root_path=data_root_path,
        image_feature_cache_root_path=image_feature_cache_root_path,
        common_config=common_config,
    )
