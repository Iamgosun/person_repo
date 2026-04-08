from __future__ import annotations

from typing import Any

from bayesvlm.data.dataset_ops import unwrap_dataset_and_indices


def _extract_base_sample_keys_and_paths(ds: Any) -> tuple[list[str], list[str] | None]:
    """
    不触发图片读取 / transform，只基于底层数据结构生成稳定 key。
    """
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


class RepeatedAugmentedFewshotDataset:
    """
    对 few-shot 训练集做“重复访问”，从而在随机增强 transform 下生成多视图缓存。
    - repeats=20 表示每个 few-shot 样本会被访问 20 次
    - sample_keys 会附加 ::aug{rep}，保证 cache manifest 与样本数一致
    """

    def __init__(self, base_ds: Any, repeats: int = 20):
        self.base_ds = base_ds
        self.repeats = int(max(repeats, 1))

        base_sample_keys, base_image_paths = _extract_base_sample_keys_and_paths(base_ds)

        self.sample_keys: list[str] = []
        self.image_paths: list[str] | None = [] if base_image_paths is not None else None

        for rep in range(self.repeats):
            for i, key in enumerate(base_sample_keys):
                self.sample_keys.append(f"{key}::aug{rep}")
                if self.image_paths is not None:
                    self.image_paths.append(str(base_image_paths[i]))

        # 尽量保留类别字段，便于外部兼容
        for attr in ("classes", "_label_names", "label_names", "classnames"):
            if hasattr(base_ds, attr):
                setattr(self, attr, list(getattr(base_ds, attr)))

    def __len__(self) -> int:
        return len(self.base_ds) * self.repeats

    def __getitem__(self, idx: int):
        base_idx = idx % len(self.base_ds)
        return self.base_ds[base_idx]