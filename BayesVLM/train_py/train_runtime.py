from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.lr_scheduler import _LRScheduler


class _BaseWarmupScheduler(_LRScheduler):
    """
    与旧版 text_only_bayes_coop 训练脚本保持一致的 warmup 包装器。
    warmup 阶段使用固定学习率；结束后把 step 权交给 successor。
    """

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch: int,
        last_epoch: int = -1,
    ):
        self.successor = successor
        self.warmup_epoch = int(warmup_epoch)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            if epoch is None:
                self.successor.step(None)
            else:
                self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):
    """
    前 warmup_epoch 个 epoch 固定学习率，之后切到后继 scheduler。
    """

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch: int,
        cons_lr: float,
        last_epoch: int = -1,
    ):
        self.cons_lr = float(cons_lr)
        super().__init__(optimizer, successor, warmup_epoch, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


def _normalize_optimizer_name(name: str | None, default_name: str) -> str:
    key = str(name or default_name).strip().lower()
    if key not in {"sgd", "adam", "adamw"}:
        raise ValueError(f"未知 optimizer: {name}，可选值为 ['sgd', 'adam', 'adamw']")
    return key


def _normalize_scheduler_name(name: str | None, default_name: str) -> str:
    key = str(name or default_name).strip().lower()
    aliases = {
        "off": "none",
        "false": "none",
    }
    key = aliases.get(key, key)
    if key not in {"none", "cosine"}:
        raise ValueError(f"未知 lr_scheduler: {name}，可选值为 ['none', 'cosine']")
    return key


def _materialize_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    params = [p for p in parameters if p.requires_grad]
    if len(params) == 0:
        raise ValueError("当前方法没有可训练参数，无法构建 optimizer。")
    return params



def build_optimizer_from_args(
    parameters: Iterable[torch.nn.Parameter],
    args,
    default_name: str,
    allow_empty: bool = False,
):
    params = [p for p in parameters if p.requires_grad]
    if len(params) == 0:
        if allow_empty:
            return None
        raise ValueError("当前方法没有可训练参数，无法构建 optimizer。")

    optimizer_name = _normalize_optimizer_name(getattr(args, "optimizer", None), default_name)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            momentum=float(getattr(args, "momentum", 0.9)),
            nesterov=bool(getattr(args, "nesterov", False)),
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

    return torch.optim.AdamW(
        params,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )





def resolve_optimizer_name(args, default_name: str) -> str:
    return _normalize_optimizer_name(getattr(args, "optimizer", None), default_name)


def build_scheduler_from_args(
    optimizer,
    args,
    default_name: str,
):
    if optimizer is None:
        return None

    scheduler_name = _normalize_scheduler_name(getattr(args, "lr_scheduler", None), default_name)

    if scheduler_name == "none":
        return None

    successor = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
    )

    warmup_epoch = int(getattr(args, "warmup_epoch", 0))
    if warmup_epoch > 0:
        return ConstantWarmupScheduler(
            optimizer=optimizer,
            successor=successor,
            warmup_epoch=warmup_epoch,
            cons_lr=float(getattr(args, "warmup_cons_lr", 1e-5)),
        )

    return successor



def resolve_scheduler_name(args, default_name: str) -> str:
    return _normalize_scheduler_name(getattr(args, "lr_scheduler", None), default_name)


def resolve_selection_metric(args, default_metric: str) -> str:
    metric = str(getattr(args, "selection_metric", None) or default_metric).strip().lower()
    if metric not in {"loss", "acc", "nlpd", "ece"}:
        raise ValueError(f"未知 selection_metric: {metric}")
    return metric


def resolve_selection_mode(args, metric: str, default_mode: str = "auto") -> str:
    mode = str(getattr(args, "selection_mode", default_mode) or default_mode).strip().lower()
    if mode == "auto":
        return "max" if metric == "acc" else "min"
    if mode not in {"min", "max"}:
        raise ValueError(f"未知 selection_mode: {mode}")
    return mode


def extract_metric_value(metrics: dict[str, float], metric: str) -> float:
    if metric not in metrics:
        raise KeyError(f"验证指标中缺少 {metric}，当前可选键：{sorted(metrics.keys())}")
    return float(metrics[metric])


def is_better_metric(current: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best
    if mode == "min":
        return current < best
    raise ValueError(f"未知 selection_mode: {mode}")


def get_current_lr(optimizer, scheduler=None) -> float:
    if optimizer is None:
        return 0.0
    if scheduler is not None:
        lrs = scheduler.get_last_lr()
        if len(lrs) > 0:
            return float(lrs[0])
    return float(optimizer.param_groups[0]["lr"])