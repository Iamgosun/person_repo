from __future__ import annotations

from pathlib import Path

import torch

from bayesvlm.training.io import save_csv, save_json, save_jsonl
from bayesvlm.training.metrics import build_official_bayesadapter_calibration_bins


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
    save_json(output_dir / "metrics.json", metrics)
    save_jsonl(output_dir / "predictions.jsonl", rows)
    torch.save(payload, output_dir / "predictions.pt")
    if "probs" not in payload or "labels" not in payload:
        raise KeyError(f"classification payload missing probs/labels, keys={list(payload.keys())}")
    bins = build_official_bayesadapter_calibration_bins(prediction=payload["probs"], label=payload["labels"], n_bins=10)
    save_csv(output_dir / "calibration_bins.csv", bins)
    return {
        "split_name": split_name,
        "metrics": metrics,
        "output_dir": str(output_dir),
        "metrics_path": str((output_dir / 'metrics.json').as_posix()),
        "predictions_pt": str((output_dir / 'predictions.pt').as_posix()),
        "predictions_jsonl": str((output_dir / 'predictions.jsonl').as_posix()),
        "bins_path": str((output_dir / 'calibration_bins.csv').as_posix()),
        "payload": payload,
    }
