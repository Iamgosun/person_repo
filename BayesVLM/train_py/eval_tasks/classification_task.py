from __future__ import annotations

from pathlib import Path

import torch

from bayesvlm.training.io import save_csv, save_json, save_jsonl
from bayesvlm.training.metrics import (
    build_official_bayesadapter_calibration_bins,
    build_adaptive_calibration_bins,
    build_selective_coverage_rows,
)


def run_classification_split(*, family, state, loader, class_names: list[str], ctx, args, split_name: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = family.evaluate_split(state, loader, class_names, ctx, args)
    rows, payload = family.collect_predictions(
        state=state,
        loader=loader,
        class_names=class_names,
        ctx=ctx,
        args=args,
        split_name=split_name,
        topk=args.prediction_topk,
    )

    save_jsonl(output_dir / "predictions.jsonl", rows)
    torch.save(payload, output_dir / "predictions.pt")

    if "probs" not in payload or "labels" not in payload:
        raise KeyError(f"classification payload missing probs/labels, keys={list(payload.keys())}")

    probs = payload["probs"]
    labels = payload["labels"]

    bins = build_official_bayesadapter_calibration_bins(
        prediction=probs,
        label=labels,
        n_bins=10,
    )
    adaptive_bins = build_adaptive_calibration_bins(
        prediction=probs,
        label=labels,
        n_bins=10,
    )
    selective_rows = build_selective_coverage_rows(
        prediction=probs,
        label=labels,
        num_classes=len(class_names),
    )

    save_json(output_dir / "metrics.json", metrics)
    save_csv(output_dir / "calibration_bins.csv", bins)
    save_csv(output_dir / "adaptive_calibration_bins.csv", adaptive_bins)
    save_csv(output_dir / "selective_metrics.csv", selective_rows)

    return {
        "split_name": split_name,
        "metrics": metrics,
        "output_dir": str(output_dir),
        "metrics_path": str((output_dir / "metrics.json").as_posix()),
        "predictions_pt": str((output_dir / "predictions.pt").as_posix()),
        "predictions_jsonl": str((output_dir / "predictions.jsonl").as_posix()),
        "bins_path": str((output_dir / "calibration_bins.csv").as_posix()),
        "adaptive_bins_path": str((output_dir / "adaptive_calibration_bins.csv").as_posix()),
        "selective_metrics_path": str((output_dir / "selective_metrics.csv").as_posix()),
        "payload": payload,
    }