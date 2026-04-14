from __future__ import annotations

import torch
import torch.distributions as dists


DEFAULT_CONF_THRESHOLDS = (0.99, 0.95, 0.90, 0.85, 0.80)


@torch.no_grad()
def _prepare_prediction_tensors(
    prediction: torch.Tensor,
    label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = prediction.detach().cpu()
    label = label.detach().cpu()

    confidence, pred_cls = prediction.max(dim=1)
    correct = (pred_cls == label).to(torch.float32)
    return confidence, pred_cls, correct


@torch.no_grad()
def calculate_official_bayesadapter_ece(
    prediction: torch.Tensor,
    label: torch.Tensor,
    n_bins: int = 10,
) -> float:
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
def build_official_bayesadapter_calibration_bins(
    prediction: torch.Tensor,
    label: torch.Tensor,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    prediction = prediction.detach().cpu()
    label = label.detach().cpu()

    confidence = prediction.max(dim=1)[0]
    pred_cls = prediction.argmax(dim=1)
    accuracies = (pred_cls == label).to(torch.float32)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    rows: list[dict[str, float | int]] = []

    for i in range(n_bins):
        bin_lower = float(bin_boundaries[i].item())
        bin_upper = float(bin_boundaries[i + 1].item())

        in_bin = confidence.gt(bin_lower) * confidence.le(bin_upper)
        count = int(in_bin.sum().item())

        if count == 0:
            rows.append(
                {
                    "bin_index": i,
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "count": 0,
                    "avg_confidence": float("nan"),
                    "avg_accuracy": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue

        avg_confidence = float(confidence[in_bin].mean().item())
        avg_accuracy = float(accuracies[in_bin].mean().item())
        rows.append(
            {
                "bin_index": i,
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "count": count,
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "gap": abs(avg_confidence - avg_accuracy),
            }
        )

    return rows


@torch.no_grad()
def calculate_adaptive_ece(
    prediction: torch.Tensor,
    label: torch.Tensor,
    n_bins: int = 10,
) -> float:
    confidence, _, correct = _prepare_prediction_tensors(prediction, label)

    order = torch.argsort(confidence)
    confidence = confidence[order]
    correct = correct[order]

    n = confidence.numel()
    if n == 0:
        return 0.0

    base = n // n_bins
    rem = n % n_bins

    aece = 0.0
    start = 0
    for i in range(n_bins):
        bin_size = base + (1 if i < rem else 0)
        if bin_size == 0:
            continue
        end = start + bin_size

        conf_bin = confidence[start:end]
        corr_bin = correct[start:end]

        aece += abs(conf_bin.mean().item() - corr_bin.mean().item()) * (bin_size / n)
        start = end

    return float(aece)


@torch.no_grad()
def build_adaptive_calibration_bins(
    prediction: torch.Tensor,
    label: torch.Tensor,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    confidence, _, correct = _prepare_prediction_tensors(prediction, label)

    order = torch.argsort(confidence)
    confidence = confidence[order]
    correct = correct[order]

    n = confidence.numel()
    if n == 0:
        return []

    base = n // n_bins
    rem = n % n_bins

    rows = []
    start = 0
    for i in range(n_bins):
        bin_size = base + (1 if i < rem else 0)
        if bin_size == 0:
            continue
        end = start + bin_size

        conf_bin = confidence[start:end]
        corr_bin = correct[start:end]

        rows.append(
            {
                "bin_index": i,
                "count": int(bin_size),
                "bin_lower": float(conf_bin.min().item()),
                "bin_upper": float(conf_bin.max().item()),
                "avg_confidence": float(conf_bin.mean().item()),
                "avg_accuracy": float(corr_bin.mean().item()),
                "gap": abs(float(conf_bin.mean().item()) - float(corr_bin.mean().item())),
            }
        )
        start = end

    return rows


@torch.no_grad()
def build_selective_coverage_rows(
    prediction: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    thresholds: tuple[float, ...] = DEFAULT_CONF_THRESHOLDS,
) -> list[dict[str, float | int | bool]]:
    confidence, _, correct = _prepare_prediction_tensors(prediction, label)

    rows = []
    total = confidence.numel()

    for thr in thresholds:
        selected = confidence >= thr
        selected_count = int(selected.sum().item())
        raw_coverage = float(selected.float().mean().item()) if total > 0 else 0.0

        if selected_count == 0:
            rows.append(
                {
                    "threshold": float(thr),
                    "selected_count": 0,
                    "raw_coverage": 0.0,
                    "selected_accuracy": 0.0,
                    "reliable": False,
                    "coverage": 0.0,
                    "covered_classes": 0,
                    "classwise_coverage": 0.0,
                }
            )
            continue

        selected_accuracy = float(correct[selected].mean().item())
        reliable = selected_accuracy >= thr

        covered_classes = int(label[selected].unique().numel())
        classwise_coverage = (covered_classes / num_classes) if reliable and num_classes > 0 else 0.0
        coverage = raw_coverage if reliable else 0.0

        rows.append(
            {
                "threshold": float(thr),
                "selected_count": selected_count,
                "raw_coverage": raw_coverage,
                "selected_accuracy": selected_accuracy,
                "reliable": bool(reliable),
                "coverage": float(coverage),
                "covered_classes": covered_classes,
                "classwise_coverage": float(classwise_coverage),
            }
        )

    return rows


@torch.no_grad()
def evaluate_prediction(
    prediction: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    pred_cls = prediction.argmax(dim=1)
    acc = (pred_cls == label).float().mean().item()
    nlpd = -dists.Categorical(prediction).log_prob(label).mean().item()
    ece = calculate_official_bayesadapter_ece(prediction, label, n_bins=10)
    aece = calculate_adaptive_ece(prediction, label, n_bins=10)

    metrics = {
        "acc": float(acc),
        "nlpd": float(nlpd),
        "ece": float(ece),
        "aece": float(aece),
    }

    selective_rows = build_selective_coverage_rows(
        prediction=prediction,
        label=label,
        num_classes=num_classes,
        thresholds=DEFAULT_CONF_THRESHOLDS,
    )
    for row in selective_rows:
        k = int(round(float(row["threshold"]) * 100))
        metrics[f"raw_cov_{k}"] = float(row["raw_coverage"])
        metrics[f"sel_acc_{k}"] = float(row["selected_accuracy"])
        metrics[f"reliable_{k}"] = 1.0 if bool(row["reliable"]) else 0.0
        metrics[f"cov_{k}"] = float(row["coverage"])
        metrics[f"class_cov_{k}"] = float(row["classwise_coverage"])

    return metrics


def make_metric_dict(loss: float, **metrics) -> dict[str, float]:
    out = {"loss": float(loss)}
    for k, v in metrics.items():
        out[k] = float(v)
    return out