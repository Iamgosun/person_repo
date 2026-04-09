from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.data.factory import SUPPORTED_MODULES
from bayesvlm.training.history import flatten_metrics_history
from bayesvlm.training.io import save_csv, save_json, tee_output
from bayesvlm.training.runtime import ensure_run_dir, set_seed
from train_py.common_experiment import build_common_context
from train_py.recipes import build_recipe
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


def _ensure_common_flags(args) -> None:
    if not hasattr(args, "cache_image_features"):
        args.cache_image_features = not getattr(args, "disable_cache_image_features", False)
    if not hasattr(args, "rebuild_image_feature_cache"):
        args.rebuild_image_feature_cache = False
    if not hasattr(args, "prediction_topk"):
        args.prediction_topk = 5
    if not hasattr(args, "recipe_name"):
        raise ValueError("args.recipe_name 未设置。")
    if not hasattr(args, "method_name") or not args.method_name:
        args.method_name = args.recipe_name
    if not hasattr(args, "model_selection"):
        args.model_selection = "best"


def _print_run_header(args, run_dir: Path, optimizer_name: str, scheduler_name: str) -> None:
    print(f"[run] recipe={args.recipe_name}")
    print(f"[run] method={args.method_name}")
    print(f"[run] dataset={args.dataset}")
    print(f"[run] model={args.model}")
    if hasattr(args, "adapter_name"):
        print(f"[run] adapter={args.adapter_name}")
    if hasattr(args, "initialization"):
        print(f"[run] initialization={args.initialization}")
    print(f"[run] shots_per_class={args.shots_per_class}")
    print(f"[run] seed={args.seed}")
    print(f"[run] device={args.device}")
    print(f"[run] run_dir={run_dir}")
    print(f"[run] optimizer={optimizer_name}")
    print(f"[run] lr_scheduler={scheduler_name}")
    print(f"[run] model_selection={args.model_selection}")
    print(f"[run] cache_image_features={args.cache_image_features}")
    print(f"[run] image_feature_cache_root={args.image_feature_cache_root}")
    print(f"[run] rebuild_image_feature_cache={args.rebuild_image_feature_cache}")


def run_recipe_from_args(args) -> None:
    _ensure_common_flags(args)

    if args.model not in MODEL_NAME_MAP:
        raise ValueError(f"无效模型名：{args.model}，可选值为 {list(MODEL_NAME_MAP.keys())}")

    if args.dataset not in SUPPORTED_MODULES:
        raise ValueError(f"无效数据集：{args.dataset}，可选值为 {sorted(SUPPORTED_MODULES.keys())}")

    recipe = build_recipe(args.recipe_name)
    set_seed(args.seed)

    optimizer_name = resolve_optimizer_name(args, recipe.default_optimizer_name)
    scheduler_name = resolve_scheduler_name(args, recipe.default_scheduler_name)
    selection_metric = resolve_selection_metric(args, recipe.default_selection_metric)
    selection_mode = resolve_selection_mode(args, selection_metric, recipe.default_selection_mode)

    run_dir = ensure_run_dir(
        save_dir=args.save_dir,
        method_name=args.method_name,
        dataset=args.dataset,
        seed=args.seed,
        path_parts=recipe.run_path_parts(args),
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_output(run_dir / "train.log"):
        run_start_time = time.time()

        _print_run_header(args, run_dir, optimizer_name, scheduler_name)
        recipe.validate_and_note(args)

        ctx = build_common_context(
            args=args,
            run_dir=run_dir,
            require_image_feature_cache=recipe.require_image_feature_cache,
        )

        state = recipe.build_state(ctx, args)
        optimizer = state["optimizer"]
        scheduler = build_scheduler_from_args(
            optimizer=optimizer,
            args=args,
            default_name=recipe.default_scheduler_name,
        )
        state["scheduler"] = scheduler

        config = {
            "recipe_name": args.recipe_name,
            "method_name": args.method_name,
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
            **recipe.build_config_extra(state, ctx, args),
        }
        save_json(run_dir / "config.json", config)

        print("[train] 开始训练 ...")
        best_score = None
        best_state = None
        best_epoch = None
        best_val_metrics = None
        last_state = None
        last_epoch = None
        last_val_metrics = None
        metrics_history = []

        for epoch in range(1, args.epochs + 1):
            train_row = recipe.train_one_epoch(state, ctx, args, epoch)

            if scheduler is not None:
                scheduler.step()

            val_metrics = recipe.evaluate_split(state, ctx.val_loader, ctx, args)
            current_score = extract_metric_value(val_metrics, selection_metric)
            current_lr = get_current_lr(optimizer, scheduler)

            row = {
                "epoch": epoch,
                "lr": current_lr,
                **train_row,
                "val": val_metrics,
            }
            metrics_history.append(row)

            print(recipe.format_epoch_log(row, ctx, args))

            if is_better_metric(current_score, best_score, selection_mode):
                best_score = current_score
                best_epoch = epoch
                best_val_metrics = val_metrics
                best_state = recipe.build_best_state(
                    state=state,
                    ctx=ctx,
                    args=args,
                    epoch=epoch,
                    val_metrics=val_metrics,
                )
                best_state["_checkpoint_epoch"] = epoch
                best_state["_checkpoint_val_metrics"] = val_metrics
                best_state["_selection_metric"] = selection_metric
                best_state["_selection_metric_value"] = current_score
                torch.save(best_state, run_dir / recipe.best_checkpoint_filename)

            last_epoch = epoch
            last_val_metrics = val_metrics
            last_state = recipe.build_best_state(
                state=state,
                ctx=ctx,
                args=args,
                epoch=epoch,
                val_metrics=val_metrics,
            )
            last_state["_checkpoint_epoch"] = epoch
            last_state["_checkpoint_val_metrics"] = val_metrics
            torch.save(last_state, run_dir / recipe.last_checkpoint_filename)
            

            if args.recipe_name == "text_only_bayes_coop":
                proto_dir = run_dir / "prototype_history"
                proto_dir.mkdir(parents=True, exist_ok=True)

                model = state["model"]
                model.eval()

                with torch.no_grad():
                    mu, _, _, _ = model.compute_text_statistics()

                num_keep = min(10, mu.shape[0])

                proto_payload = {
                    "epoch": epoch,
                    "class_ids": list(range(num_keep)),
                    "class_names": ctx.class_names[:num_keep],
                    "prototypes": mu[:num_keep].detach().cpu(),
                }

                torch.save(proto_payload, proto_dir / f"epoch_{epoch:03d}.pt")



            save_json(run_dir / "metrics_history.json", metrics_history)
            save_csv(run_dir / "metrics_history.csv", flatten_metrics_history(metrics_history))





        if best_state is None:
            raise RuntimeError("训练未产生 best checkpoint")

        if args.model_selection == "best":
            selected_label = "best"
            selected_ckpt_path = run_dir / recipe.best_checkpoint_filename
        elif args.model_selection == "last":
            selected_label = "last"
            selected_ckpt_path = run_dir / recipe.last_checkpoint_filename
        else:
            raise ValueError(f"未知 model_selection: {args.model_selection}")

        if not selected_ckpt_path.exists():
            raise RuntimeError(f"未找到目标权重文件：{selected_ckpt_path}")

        print(f"[final] 加载 {selected_label} 权重并导出最终预测 ...")
        selected_state = torch.load(selected_ckpt_path, map_location=args.device)
        recipe.load_best_state(state, selected_state, ctx, args)

        final_train_metrics = recipe.evaluate_split(state, ctx.train_eval_loader, ctx, args)
        final_val_metrics = recipe.evaluate_split(state, ctx.val_loader, ctx, args)
        final_test_metrics = recipe.evaluate_split(state, ctx.test_loader, ctx, args)

        recipe.dump_predictions(state, ctx, args)

        summary = {
            "recipe_name": args.recipe_name,
            "method_name": args.method_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "model_selection": args.model_selection,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "selected_checkpoint": selected_label,
            "selected_epoch": selected_state.get("_checkpoint_epoch"),
            "selected_val_metrics": selected_state.get("_checkpoint_val_metrics"),
            "best_epoch": best_epoch,
            "best_val_metrics_saved": best_val_metrics,
            "last_epoch": last_epoch,
            "last_val_metrics_saved": last_val_metrics,
            "final_train_metrics_recomputed": final_train_metrics,
            "final_val_metrics_recomputed": final_val_metrics,
            "final_test_metrics_recomputed": final_test_metrics,
            "run_dir": str(run_dir),
            "elapsed_seconds": round(time.time() - run_start_time, 2),
            "artifacts": {
                "log_file": "train.log",
                "config_file": "config.json",
                "class_names_file": "class_names.json",
                "metrics_json_file": "metrics_history.json",
                "metrics_csv_file": "metrics_history.csv",
                "best_ckpt_file": recipe.best_checkpoint_filename,
                "last_ckpt_file": recipe.last_checkpoint_filename,
                "train_predictions_jsonl": "train_predictions.jsonl",
                "train_predictions_pt": "train_predictions.pt",
                "val_predictions_jsonl": "val_predictions.jsonl",
                "val_predictions_pt": "val_predictions.pt",
                "test_predictions_jsonl": "test_predictions.jsonl",
                "test_predictions_pt": "test_predictions.pt",
            },
        }
        summary.update(recipe.build_summary_extra(state, best_state, ctx, args))
        save_json(run_dir / "summary.json", summary)

        print("[done] 最终结果：")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
