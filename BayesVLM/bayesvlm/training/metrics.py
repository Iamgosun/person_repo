from __future__ import annotations

import torch
import torch.distributions as dists


@torch.no_grad()
def calculate_official_bayesadapter_ece(
    prediction: torch.Tensor,
    label: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    复现官方 BayesAdapter utils.py 的 ECE 口径：
    - 输入 prediction: [N, C]，为概率分布
    - confidence = MSP = prediction.max(dim=1)[0]
    - correctness = top-1 预测是否正确
    - 10 个等宽 bin
    """
    prediction = prediction.detach().cpu()
    label = label.detach().cpu()

    confidence = prediction.max(dim=1)[0]
    is_misclassif = (prediction.argmax(dim=1) != label).to(torch.int)
    accuracies = 1 - is_misclassif

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.tensor(0.0)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidence.gt(bin_lower.item()) * confidence.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            ece = ece + torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece.item())


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

    ece 使用官方 BayesAdapter 的 10-bin MSP ECE
    """
    del num_classes

    pred_cls = prediction.argmax(dim=1)
    acc = (pred_cls == label).float().mean().item()
    nlpd = -dists.Categorical(prediction).log_prob(label).mean().item()
    ece = calculate_official_bayesadapter_ece(prediction, label, n_bins=10)
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