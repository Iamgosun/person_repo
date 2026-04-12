from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.data.factory import SUPPORTED_MODULES
from bayesvlm.training.history import flatten_metrics_history
from bayesvlm.training.io import save_csv, save_json, tee_output
from bayesvlm.training.runtime import set_seed
from train_py.families import build_family
from train_py.protocols import build_protocol
from train_py.runtime.common_context import build_common_context
from train_py.runtime.experiment_paths import build_run_dir
from train_py.train_runtime import (
    build_scheduler_from_args,
    extract_metric_value,
    get_current_lr,
    is_better_metric,
    resolve_optimizer_name,
    resolve_scheduler_name,
    resolve_selection_metric,
    resolve_selection_mode,
)


def run_family_train(args) -> None:
    if args.model not in MODEL_NAME_MAP:
        raise ValueError(f"invalid model: {args.model}, choices: {list(MODEL_NAME_MAP.keys())}")
    if args.dataset not in SUPPORTED_MODULES:
        raise ValueError(f"invalid dataset: {args.dataset}, choices: {sorted(SUPPORTED_MODULES.keys())}")

    family = build_family(args.family)
    protocol = build_protocol(args.protocol)
    set_seed(args.seed)

    args._family_run_path_parts = family.run_path_parts(args)
    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 训练阶段会显式写这些目录，必须提前建好
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "train").mkdir(parents=True, exist_ok=True)

    optimizer_name = resolve_optimizer_name(args, family.default_optimizer_name)

    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    optimizer_name = resolve_optimizer_name(args, family.default_optimizer_name)
    scheduler_name = resolve_scheduler_name(args, family.default_scheduler_name)
    selection_metric = resolve_selection_metric(args, family.default_selection_metric)
    selection_mode = resolve_selection_mode(args, selection_metric, family.default_selection_mode)

    with tee_output(run_dir / "train" / "train.log"):
        run_start_time = time.time()
        print(f"[run] family={args.family}")
        print(f"[run] variant={args.variant}")
        print(f"[run] protocol={args.protocol}")
        print(f"[run] dataset={args.dataset}")
        print(f"[run] model={args.model}")
        print(f"[run] shots_per_class={args.shots_per_class}")
        print(f"[run] seed={args.seed}")
        print(f"[run] device={args.device}")
        print(f"[run] run_dir={run_dir}")
        print(f"[run] optimizer={optimizer_name}")
        print(f"[run] lr_scheduler={scheduler_name}")
        print(f"[run] model_selection={args.model_selection}")

        family.validate_and_note(args)
        prepared = protocol.prepare_train_data(args)
        ctx = build_common_context(args=args, run_dir=run_dir, prepared=prepared, require_image_feature_cache=family.require_image_feature_cache)
        state = family.build_state(ctx, args)
        optimizer = state["optimizer"]
        scheduler = build_scheduler_from_args(optimizer=optimizer, args=args, default_name=family.default_scheduler_name)
        state["scheduler"] = scheduler

        config = {
            **vars(args),
            **ctx.common_config,
            "optimizer": optimizer_name,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "lr_scheduler": scheduler_name,
            "warmup_epoch": getattr(args, "warmup_epoch", 0),
            "warmup_cons_lr": getattr(args, "warmup_cons_lr", 1e-5),
            "momentum": getattr(args, "momentum", 0.9),
            "nesterov": getattr(args, "nesterov", False),
            "model_selection": args.model_selection,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "run_dir": str(run_dir),
            **family.build_config_extra(state, ctx, args),
            **protocol.build_summary_extra(ctx, args),
        }
        save_json(run_dir / "config" / "config.json", config)

        print("[train] starting training loop ...")
        best_score = None
        best_state = None
        best_epoch = None
        best_val_metrics = None
        last_state = None
        last_epoch = None
        last_val_metrics = None
        metrics_history = []

        if int(args.epochs) == 0:
            init_val_metrics = family.evaluate_split(state, ctx.val_loader, ctx.class_names, ctx, args)
            init_score = extract_metric_value(init_val_metrics, selection_metric)
            current_lr = get_current_lr(optimizer, scheduler)
            row = {"epoch": 0, "lr": current_lr, "init_only": True, "val": init_val_metrics}
            metrics_history.append(row)
            best_score = init_score
            best_epoch = 0
            best_val_metrics = init_val_metrics
            best_state = family.build_best_state(state=state, ctx=ctx, args=args, epoch=0, val_metrics=init_val_metrics)
            best_state["_checkpoint_epoch"] = 0
            best_state["_checkpoint_val_metrics"] = init_val_metrics
            best_state["_selection_metric"] = selection_metric
            best_state["_selection_metric_value"] = init_score
            torch.save(best_state, run_dir / "checkpoints" / family.best_checkpoint_filename)
            last_epoch = 0
            last_val_metrics = init_val_metrics
            last_state = family.build_best_state(state=state, ctx=ctx, args=args, epoch=0, val_metrics=init_val_metrics)
            last_state["_checkpoint_epoch"] = 0
            last_state["_checkpoint_val_metrics"] = init_val_metrics
            torch.save(last_state, run_dir / "checkpoints" / family.last_checkpoint_filename)
            save_csv(run_dir / "train" / "metrics_history.csv", flatten_metrics_history(metrics_history))
        else:
            for epoch in range(1, args.epochs + 1):
                train_row = family.train_one_epoch(state, ctx, args, epoch)
                if scheduler is not None:
                    scheduler.step()
                val_metrics = family.evaluate_split(state, ctx.val_loader, ctx.class_names, ctx, args)
                current_score = extract_metric_value(val_metrics, selection_metric)
                current_lr = get_current_lr(optimizer, scheduler)
                row = {"epoch": epoch, "lr": current_lr, **train_row, "val": val_metrics}
                metrics_history.append(row)
                print(family.format_epoch_log(row, ctx, args))
                if is_better_metric(current_score, best_score, selection_mode):
                    best_score = current_score
                    best_epoch = epoch
                    best_val_metrics = val_metrics
                    best_state = family.build_best_state(state=state, ctx=ctx, args=args, epoch=epoch, val_metrics=val_metrics)
                    best_state["_checkpoint_epoch"] = epoch
                    best_state["_checkpoint_val_metrics"] = val_metrics
                    best_state["_selection_metric"] = selection_metric
                    best_state["_selection_metric_value"] = current_score
                    torch.save(best_state, run_dir / "checkpoints" / family.best_checkpoint_filename)
                last_epoch = epoch
                last_val_metrics = val_metrics
                last_state = family.build_best_state(state=state, ctx=ctx, args=args, epoch=epoch, val_metrics=val_metrics)
                last_state["_checkpoint_epoch"] = epoch
                last_state["_checkpoint_val_metrics"] = val_metrics
                torch.save(last_state, run_dir / "checkpoints" / family.last_checkpoint_filename)
                save_csv(run_dir / "train" / "metrics_history.csv", flatten_metrics_history(metrics_history))

        if best_state is None:
            raise RuntimeError("training did not produce a best checkpoint")

        train_summary = {
            "family": args.family,
            "variant": args.variant,
            "protocol": args.protocol,
            "dataset": args.dataset,
            "seed": args.seed,
            "model_selection": args.model_selection,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "best_epoch": best_epoch,
            "best_val_metrics_saved": best_val_metrics,
            "last_epoch": last_epoch,
            "last_val_metrics_saved": last_val_metrics,
            "run_dir": str(run_dir),
            "elapsed_seconds": round(time.time() - run_start_time, 2),
            "artifacts": {
                "log_file": "train/train.log",
                "config_file": "config/config.json",
                "class_names_file": "config/class_names.json",
                "metrics_csv_file": "train/metrics_history.csv",
                "best_ckpt_file": f"checkpoints/{family.best_checkpoint_filename}",
                "last_ckpt_file": f"checkpoints/{family.last_checkpoint_filename}",
            },
        }
        save_json(run_dir / "train" / "train_summary.json", train_summary)
        print("[done] training complete:")
        print(json.dumps(train_summary, ensure_ascii=False, indent=2))
