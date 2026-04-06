from __future__ import annotations

import torch
import torch.distributions as dists
from torchmetrics.classification import MulticlassCalibrationError


@torch.no_grad()
def evaluate_prediction(
    prediction: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
) -> tuple[float, float, float]:
    """
    prediction: [N, C] 概率分布
    label: [N]
    返回: (acc, nlpd, ece)
    """
    ece_metric = MulticlassCalibrationError(
        num_classes=num_classes,
        n_bins=20,
        norm="l1",
    )

    pred_cls = prediction.argmax(dim=1)
    acc = (pred_cls == label).float().mean().item()
    nlpd = -dists.Categorical(prediction).log_prob(label).mean().item()
    ece = ece_metric(prediction, label).item()
    return acc, nlpd, ece


def make_metric_dict(
    loss: float,
    acc: float,
    nlpd: float,
    ece: float,
) -> dict[str, float]:
    return {
        "loss": float(loss),
        "acc": float(acc),
        "nlpd": float(nlpd),
        "ece": float(ece),
    }