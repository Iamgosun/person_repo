from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bayesvlm.training.io import save_json
from bayesvlm.training.ood_metrics import compute_ood_metrics_from_id_scores


def _compute_id_score_from_payload(payload: dict, score_name: str) -> torch.Tensor:
    score_key = str(score_name).strip().lower()
    probs = payload["probs"].float()
    if score_key == "msp":
        return probs.max(dim=1).values
    if score_key == "entropy":
        p = probs.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=1)
        return -entropy
    if score_key == "energy":
        if "logits" in payload:
            logits = payload["logits"].float()
        elif "logits_mean" in payload:
            logits = payload["logits_mean"].float()
        else:
            raise KeyError(f"payload missing logits/logits_mean, keys={list(payload.keys())}")
        return torch.logsumexp(logits, dim=1)
    raise ValueError(f"unknown OOD score: {score_name}")


def run_ood_detection(*, family, state, id_payload: dict, ood_loaders: dict[str, Any], ctx, args, output_root: Path) -> dict:
    if not ood_loaders:
        return {"enabled": False, "targets": []}
    score_names = list(getattr(args, "ood_scores", ["msp", "entropy", "energy"]))
    summary_targets = []
    for target_name, loader in ood_loaders.items():
        ood_payload = family.collect_ood_payload(state=state, loader=loader, ctx=ctx, args=args)
        target_entry = {
            "dataset": target_name,
            "num_id_samples": int(id_payload["probs"].shape[0]),
            "num_ood_samples": int(ood_payload["probs"].shape[0]),
            "scores": [],
        }
        for score_name in score_names:
            id_scores = _compute_id_score_from_payload(id_payload, score_name)
            ood_scores = _compute_id_score_from_payload(ood_payload, score_name)
            metrics = compute_ood_metrics_from_id_scores(id_scores, ood_scores)
            metrics.update({
                "score_name": score_name,
                "score_semantics": "higher_means_more_ID_like",
                "target_dataset": target_name,
            })
            out_dir = output_root / score_name / target_name
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json(out_dir / "metrics.json", metrics)
            target_entry["scores"].append({
                "score_name": score_name,
                "metrics_path": str((out_dir / 'metrics.json').as_posix()),
            })
        summary_targets.append(target_entry)
    return {"enabled": True, "targets": summary_targets}
