from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch



def _to_numpy(x: torch.Tensor | np.ndarray | Iterable[float]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(list(x), dtype=np.float64)



def _binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    return fps.astype(np.float64), tps.astype(np.float64), thresholds.astype(np.float64)



def _safe_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return float(np.trapz(y, x))



def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    num_pos = int(y_true.sum())
    num_neg = int((1 - y_true).sum())
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    fpr = np.r_[0.0, fps / max(num_neg, 1), 1.0]
    tpr = np.r_[0.0, tps / max(num_pos, 1), 1.0]
    return _safe_auc(fpr, tpr)



def binary_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    num_pos = int(y_true.sum())
    if num_pos == 0:
        return float("nan")

    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / max(num_pos, 1)

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return _safe_auc(recall, precision)



def fpr_at_95_tpr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    num_pos = int(y_true.sum())
    num_neg = int((1 - y_true).sum())
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    fpr = fps / max(num_neg, 1)
    tpr = tps / max(num_pos, 1)

    valid = np.where(tpr >= 0.95)[0]
    if len(valid) == 0:
        return float("nan")
    return float(fpr[valid[0]])



def compute_ood_metrics_from_id_scores(id_scores, ood_scores) -> dict[str, float]:
    id_scores = _to_numpy(id_scores)
    ood_scores = _to_numpy(ood_scores)

    if id_scores.ndim != 1 or ood_scores.ndim != 1:
        raise ValueError(
            f"OOD scores 必须是一维向量，当前 shapes: id={id_scores.shape}, ood={ood_scores.shape}"
        )

    all_scores = np.concatenate([id_scores, ood_scores], axis=0)
    labels_in = np.concatenate(
        [
            np.ones_like(id_scores, dtype=np.int64),
            np.zeros_like(ood_scores, dtype=np.int64),
        ],
        axis=0,
    )
    labels_out = 1 - labels_in

    metrics = {
        "AUROC": float(binary_auroc(labels_in, all_scores)),
        "AUPR_IN": float(binary_aupr(labels_in, all_scores)),
        "AUPR_OUT": float(binary_aupr(labels_out, -all_scores)),
        "FPR95": float(fpr_at_95_tpr(labels_in, all_scores)),
        "num_id_samples": int(id_scores.shape[0]),
        "num_ood_samples": int(ood_scores.shape[0]),
        "id_score_mean": float(np.mean(id_scores)) if id_scores.size > 0 else float("nan"),
        "ood_score_mean": float(np.mean(ood_scores)) if ood_scores.size > 0 else float("nan"),
        "id_score_std": float(np.std(id_scores)) if id_scores.size > 0 else float("nan"),
        "ood_score_std": float(np.std(ood_scores)) if ood_scores.size > 0 else float("nan"),
    }
    return metrics
