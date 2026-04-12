from __future__ import annotations


def build_eval_summary_base(*, args, run_dir: str, selected_checkpoint: str, selected_epoch, selected_val_metrics, classification_results: dict, ood_result: dict, protocol_extra: dict, family_extra: dict, elapsed_seconds: float) -> dict:
    summary = {
        "family": args.family,
        "variant": args.variant,
        "protocol": args.protocol,
        "dataset": args.dataset,
        "seed": args.seed,
        "evaluation_tasks": list(args.evaluation_tasks),
        "model_selection": args.model_selection,
        "selected_checkpoint": selected_checkpoint,
        "selected_epoch": selected_epoch,
        "selected_val_metrics": selected_val_metrics,
        "classification": classification_results,
        "ood_detection": ood_result,
        "run_dir": run_dir,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    summary.update(protocol_extra)
    summary.update(family_extra)
    return summary
