from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesvlm.training.io import save_json, tee_output
from train_py.eval_tasks.classification_task import run_classification_split
from train_py.eval_tasks.ood_detection_task import run_ood_detection
from train_py.families import build_family
from train_py.protocols import build_protocol
from train_py.runtime.common_context import build_common_context
from train_py.runtime.experiment_paths import build_run_dir
from train_py.runtime.summary_builder import build_eval_summary_base


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config/config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_selected_checkpoint(run_dir: Path, family, model_selection: str) -> tuple[str, Path]:
    selection = str(model_selection).strip().lower()
    if selection == "best":
        return "best", run_dir / "checkpoints" / family.best_checkpoint_filename
    if selection == "last":
        return "last", run_dir / "checkpoints" / family.last_checkpoint_filename
    raise ValueError(f"unknown model_selection: {model_selection}")


def _resolve_loader_attr(ctx, attr_path: str):
    cur = ctx
    for part in attr_path.split('.'):
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def run_family_eval(args) -> None:
    family = build_family(args.family)
    args._family_run_path_parts = family.run_path_parts(args)
    run_dir = build_run_dir(args)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    eval_root = run_dir / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    with tee_output(eval_root / "eval.log"):
        eval_start_time = time.time()
        cfg = _load_run_config(run_dir)
        saved_args = SimpleNamespace(**cfg)
        family = build_family(saved_args.family)
        protocol = build_protocol(saved_args.protocol)
        print(f"[eval] run_dir={run_dir}")
        print(f"[eval] family={saved_args.family}")
        print(f"[eval] variant={saved_args.variant}")
        print(f"[eval] protocol={saved_args.protocol}")
        prepared = protocol.prepare_eval_data(saved_args)
        ctx = build_common_context(args=saved_args, run_dir=run_dir, prepared=prepared, require_image_feature_cache=family.require_image_feature_cache)
        state = family.build_state(ctx, saved_args)
        selected_label, selected_ckpt_path = _resolve_selected_checkpoint(run_dir, family, saved_args.model_selection)
        if not selected_ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {selected_ckpt_path}")
        print(f"[eval] loading {selected_label} checkpoint: {selected_ckpt_path}")
        selected_state = torch.load(selected_ckpt_path, map_location=saved_args.device)
        family.load_best_state(state, selected_state, ctx, saved_args)

        split_results = {}
        id_payload = None
        for split in protocol.classification_splits(ctx, saved_args):
            loader = _resolve_loader_attr(ctx, split.loader_attr)
            class_names = _resolve_loader_attr(ctx, split.class_names_attr)
            result = run_classification_split(
                family=family,
                state=state,
                loader=loader,
                class_names=class_names,
                ctx=ctx,
                args=saved_args,
                split_name=split.split_name,
                output_dir=run_dir / split.relative_output_dir,
            )
            split_results[split.split_name] = {k: v for k, v in result.items() if k != "payload"}
            if split.split_name == getattr(saved_args, "ood_reference_split", "test"):
                id_payload = result["payload"]

        ood_result = {"enabled": False, "targets": []}
        if "ood_detection" in saved_args.evaluation_tasks:
            if id_payload is None:
                raise ValueError(
                    f"ood_reference_split={saved_args.ood_reference_split} was not produced by protocol classification_splits"
                )
            ood_loaders = {}
            for target_name in getattr(saved_args, "ood_datasets", []):
                if target_name in ctx.extra_eval_loaders:
                    ood_loaders[target_name] = ctx.extra_eval_loaders[target_name]
            ood_result = run_ood_detection(
                family=family,
                state=state,
                id_payload=id_payload,
                ood_loaders=ood_loaders,
                ctx=ctx,
                args=saved_args,
                output_root=run_dir / "eval" / "ood",
            )

        family_extra = family.build_summary_extra(state, selected_state, ctx, saved_args)
        protocol_extra = protocol.build_summary_extra(ctx, saved_args)
        summary = build_eval_summary_base(
            args=saved_args,
            run_dir=str(run_dir),
            selected_checkpoint=selected_label,
            selected_epoch=selected_state.get("_checkpoint_epoch"),
            selected_val_metrics=selected_state.get("_checkpoint_val_metrics"),
            classification_results=split_results,
            ood_result=ood_result,
            protocol_extra=protocol_extra,
            family_extra=family_extra,
            elapsed_seconds=time.time() - eval_start_time,
        )
        save_json(run_dir / "summary.json", summary)
        print("[done] evaluation complete:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
