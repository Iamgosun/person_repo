from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from bayesvlm.common import EncoderResult
from bayesvlm.data.dataset_ops import unwrap_dataset_and_indices
from bayesvlm.precompute import precompute_image_features


@dataclass(frozen=True)
class ImageFeatureCacheSpec:
    dataset: str
    split: str                  # train_full / val / test / train_fewshot_aug_seedX_shotY_repZ
    model_str: str
    local_model_path: str | None
    image_size: int
    transform_name: str
    data_root: str


@dataclass
class ImageFeatureBundle:
    outputs: EncoderResult
    class_ids: torch.Tensor
    sample_keys: list[str]
    image_paths: list[str] | None = None

    def subset(self, indices: list[int] | torch.Tensor) -> "ImageFeatureBundle":
        if torch.is_tensor(indices):
            indices = indices.tolist()

        return ImageFeatureBundle(
            outputs=self.outputs[indices],
            class_ids=self.class_ids[indices],
            sample_keys=[self.sample_keys[i] for i in indices],
            image_paths=None if self.image_paths is None else [self.image_paths[i] for i in indices],
        )


def _sha256_str_list(values: list[str]) -> str:
    h = hashlib.sha256()
    for x in values:
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _safe_name(x: str | None) -> str:
    if x is None:
        return "none"
    x = str(x)
    for old, new in [
        ("/", "_"),
        ("\\", "_"),
        (" ", "_"),
        (":", "_"),
        (";", "_"),
        ("=", "_"),
    ]:
        x = x.replace(old, new)
    return x


def _short_hash(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()[:12]


def build_cache_dir(cache_root: str | Path, spec: ImageFeatureCacheSpec) -> Path:
    root = Path(cache_root)
    tfm_hash = _short_hash(spec.transform_name)

    return (
        root
        / spec.dataset
        / f"model_{_safe_name(spec.model_str)}"
        / f"weights_{_safe_name(spec.local_model_path)}"
        / f"imgsz_{spec.image_size}"
        / f"tfm_{tfm_hash}"
        / spec.split
    )


def extract_sample_keys_and_paths(ds: Any) -> tuple[list[str], list[str] | None]:
    """
    优先使用 image_path 作为跨任务稳定 key。
    如果底层数据集没有路径信息，再退化成 base-dataset 索引 key。

    新增支持：
    - 自定义 wrapper 只要暴露 sample_keys / image_paths，就直接使用
    """
    if hasattr(ds, "sample_keys"):
        sample_keys = list(ds.sample_keys)
        image_paths = list(ds.image_paths) if hasattr(ds, "image_paths") and ds.image_paths is not None else None
        return sample_keys, image_paths

    base_ds, base_indices = unwrap_dataset_and_indices(ds)

    if hasattr(base_ds, "_samples"):
        src = base_ds._samples
        idxs = base_indices if base_indices is not None else list(range(len(src)))
        image_paths = [str(src[i][0]) for i in idxs]
        return image_paths, image_paths

    if hasattr(base_ds, "_split_info"):
        src = base_ds._split_info
        idxs = base_indices if base_indices is not None else list(range(len(src)))
        image_paths = [str(src[i][0]) for i in idxs]
        return image_paths, image_paths

    idxs = base_indices if base_indices is not None else list(range(len(base_ds)))
    keys = [f"{base_ds.__class__.__name__}:{i}" for i in idxs]
    return keys, None


def _cache_file_paths(cache_dir: Path) -> list[Path]:
    return [
        cache_dir / "embeddings_img.pt",
        cache_dir / "activations_img.pt",
        cache_dir / "residuals_img.pt",
        cache_dir / "class_ids_img.pt",
        cache_dir / "image_ids.pt",
        cache_dir / "sample_keys.json",
        cache_dir / "image_paths.json",
        cache_dir / "manifest.json",
    ]


def _required_core_cache_files(cache_dir: Path) -> list[Path]:
    return [
        cache_dir / "embeddings_img.pt",
        cache_dir / "activations_img.pt",
        cache_dir / "residuals_img.pt",
        cache_dir / "class_ids_img.pt",
        cache_dir / "image_ids.pt",
        cache_dir / "sample_keys.json",
        cache_dir / "manifest.json",
    ]


def _manifest_diff_keys(current_manifest: dict, expected_manifest: dict) -> list[str]:
    keys = sorted(set(current_manifest.keys()) | set(expected_manifest.keys()))
    return [k for k in keys if current_manifest.get(k) != expected_manifest.get(k)]


def clear_feature_cache(cache_dir: str | Path) -> None:
    cache_dir = Path(cache_dir)
    for p in _cache_file_paths(cache_dir):
        if p.exists():
            p.unlink()


def load_image_feature_bundle(cache_dir: str | Path) -> ImageFeatureBundle:
    cache_dir = Path(cache_dir)

    outputs = EncoderResult(
        embeds=torch.load(cache_dir / "embeddings_img.pt", map_location="cpu"),
        activations=torch.load(cache_dir / "activations_img.pt", map_location="cpu"),
        residuals=torch.load(cache_dir / "residuals_img.pt", map_location="cpu"),
    )
    class_ids = torch.load(cache_dir / "class_ids_img.pt", map_location="cpu")

    sample_keys = json.loads((cache_dir / "sample_keys.json").read_text(encoding="utf-8"))

    image_paths = None
    image_paths_path = cache_dir / "image_paths.json"
    if image_paths_path.exists():
        image_paths = json.loads(image_paths_path.read_text(encoding="utf-8"))

    return ImageFeatureBundle(
        outputs=outputs,
        class_ids=class_ids,
        sample_keys=sample_keys,
        image_paths=image_paths,
    )


def get_or_build_image_feature_bundle(
    *,
    image_encoder,
    loader,
    ds,
    cache_root: str | Path,
    spec: ImageFeatureCacheSpec,
    force_rebuild: bool = False,
) -> ImageFeatureBundle:
    cache_dir = build_cache_dir(cache_root, spec)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"
    sample_keys, image_paths = extract_sample_keys_and_paths(ds)

    expected_manifest = {
        **asdict(spec),
        "num_samples": len(sample_keys),
        "sample_keys_sha256": _sha256_str_list(sample_keys),
        "feature_format_version": 2,
    }

    required_files = _required_core_cache_files(cache_dir)
    missing_core_files = [str(p.name) for p in required_files if not p.exists()]
    has_core_files = len(missing_core_files) == 0

    print(
        "[cache] "
        f"dataset={spec.dataset} split={spec.split} "
        f"cache_dir={cache_dir}"
    )

    if force_rebuild:
        print("[cache] force_rebuild=True，忽略已有缓存并重建。")
    elif has_core_files:
        current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current_manifest == expected_manifest:
            print("[cache] 命中缓存，直接加载。")
            return load_image_feature_bundle(cache_dir)

        diff_keys = _manifest_diff_keys(current_manifest, expected_manifest)
        print("[cache] manifest 不匹配，准备重建。")
        print(f"[cache] manifest_diff_keys={diff_keys}")
        for k in diff_keys:
            print(f"[cache]   {k}: current={current_manifest.get(k)} | expected={expected_manifest.get(k)}")
    else:
        print("[cache] 未命中缓存，缺少核心文件。")
        print(f"[cache] missing_core_files={missing_core_files}")

    # 只要 manifest 不匹配，或核心文件不完整，就先清旧缓存。
    # 否则 precompute_image_features() 自己也会误命中旧 pt 文件。
    clear_feature_cache(cache_dir)

    outputs, class_ids, _ = precompute_image_features(
        image_encoder=image_encoder,
        loader=loader,
        save_predictions=True,
        cache_dir=cache_dir,
    )

    (cache_dir / "sample_keys.json").write_text(
        json.dumps(sample_keys, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if image_paths is not None:
        (cache_dir / "image_paths.json").write_text(
            json.dumps(image_paths, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manifest_path.write_text(
        json.dumps(expected_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[cache] 已完成重建并写入 manifest。")

    return ImageFeatureBundle(
        outputs=outputs,
        class_ids=class_ids,
        sample_keys=sample_keys,
        image_paths=image_paths,
    )