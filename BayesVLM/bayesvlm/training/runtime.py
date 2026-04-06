from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    固定随机种子，尽量保证 few-shot 抽样与训练可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_run_dir(
    save_dir: str,
    method_name: str,
    dataset: str,
    seed: int,
    path_parts: Sequence[str] | None = None,
) -> Path:
    """
    通用 run_dir 构造器。

    示例:
    - text_only_bayes_coop:
        output/text_only_bayes_coop/cifar10/shot_16/seed_42
    - vlm_adapter:
        output/vlm_adapter/cifar10/LP/RANDOM/shot_16/seed_42
    """
    parts = [save_dir, method_name, dataset]

    if path_parts:
        parts.extend([str(x) for x in path_parts])

    parts.append(f"seed_{seed}")

    run_dir = Path(parts[0])
    for part in parts[1:]:
        run_dir = run_dir / part

    return run_dir


def ensure_run_dir(
    save_dir: str,
    method_name: str,
    dataset: str,
    seed: int,
    path_parts: Sequence[str] | None = None,
) -> Path:
    run_dir = build_run_dir(
        save_dir=save_dir,
        method_name=method_name,
        dataset=dataset,
        seed=seed,
        path_parts=path_parts,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir