from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import torch.nn as nn


@dataclass
class ClassUncertaintySummary:
    class_names: List[str]
    trace: torch.Tensor           # [C]
    logdet: torch.Tensor          # [C]
    diag: torch.Tensor            # [C, D]
    min_eig: torch.Tensor         # [C]
    max_eig: torch.Tensor         # [C]
    topk_dim_idx: torch.Tensor    # [C, K]
    topk_dim_val: torch.Tensor    # [C, K]


@torch.no_grad()
def summarize_class_text_uncertainty(
    class_text_distributions,
    eps: float = 1e-6,
    topk: int = 10,
) -> ClassUncertaintySummary:
    """
    输入:
        class_text_distributions.covariance: [C, D, D]
        class_text_distributions.class_names: list[str]

    输出:
        每个类别的 trace / logdet / diag / 特征值范围 / 最不确定维度
    """
    cov = class_text_distributions.covariance
    class_names = class_text_distributions.class_names

    if cov.ndim != 3:
        raise ValueError(f"covariance should be [C, D, D], got {cov.shape}")

    C, D, D2 = cov.shape
    if D != D2:
        raise ValueError(f"covariance should be square, got {cov.shape}")

    # 对称化，避免数值误差
    cov = 0.5 * (cov + cov.transpose(-1, -2))

    eye = torch.eye(D, device=cov.device, dtype=cov.dtype).unsqueeze(0)  # [1, D, D]
    cov_reg = cov + eps * eye

    diag = torch.diagonal(cov, dim1=-2, dim2=-1)      # [C, D]
    trace = diag.sum(dim=-1)                          # [C]

    # logdet 用 slogdet 更稳
    sign, logabsdet = torch.linalg.slogdet(cov_reg)
    # 若 sign <= 0，说明数值上不是严格正定；这里仍保留 logabsdet 供排查
    logdet = logabsdet

    eigvals = torch.linalg.eigvalsh(cov)              # [C, D]
    min_eig = eigvals[:, 0]
    max_eig = eigvals[:, -1]

    k = min(topk, D)
    topk_dim_val, topk_dim_idx = torch.topk(diag, k=k, dim=-1)

    return ClassUncertaintySummary(
        class_names=class_names,
        trace=trace,
        logdet=logdet,
        diag=diag,
        min_eig=min_eig,
        max_eig=max_eig,
        topk_dim_idx=topk_dim_idx,
        topk_dim_val=topk_dim_val,
    )


@torch.no_grad()
def print_class_uncertainty_report(
    summary: ClassUncertaintySummary,
    topn: int | None = None,
):
    """
    按 trace 从大到小打印类别不确定性摘要
    """
    order = torch.argsort(summary.trace, descending=True)

    if topn is not None:
        order = order[:topn]

    print("\n===== Class Text Uncertainty Report =====")
    for idx in order.tolist():
        name = summary.class_names[idx]
        tr = summary.trace[idx].item()
        ld = summary.logdet[idx].item()
        mine = summary.min_eig[idx].item()
        maxe = summary.max_eig[idx].item()

        top_dims = summary.topk_dim_idx[idx].tolist()
        top_vals = [round(v, 6) for v in summary.topk_dim_val[idx].tolist()]

        print(
            f"[{name}] "
            f"trace={tr:.6f}, "
            f"logdet={ld:.6f}, "
            f"min_eig={mine:.6e}, "
            f"max_eig={maxe:.6e}, "
            f"top_uncertain_dims={top_dims}, "
            f"top_uncertain_vals={top_vals}"
        )