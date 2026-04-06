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


def _ensure_common_flags(args) -> None:
    if not hasattr(args, "cache_image_features"):
        args.cache_image_features = not getattr(args, "disable_cache_image_features", False)
    if not hasattr(args, "rebuild_image_feature_cache"):
        args.rebuild_image_feature_cache = False
    if not hasattr(args, "prediction_topk"):
        args.prediction_topk = 5
    if not hasattr(args, "method_name"):
        raise ValueError("args.method_name 未设置。")


def _print_run_header(args, run_dir: Path) -> None:
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
    print(f"[run] cache_image_features={args.cache_image_features}")
    print(f"[run] image_feature_cache_root={args.image_feature_cache_root}")
    print(f"[run] rebuild_image_feature_cache={args.rebuild_image_feature_cache}")


def run_recipe_from_args(args) -> None:
    _ensure_common_flags(args)

    if args.model not in MODEL_NAME_MAP:
        raise ValueError(f"无效模型名：{args.model}，可选值为 {list(MODEL_NAME_MAP.keys())}")

    if args.dataset not in SUPPORTED_MODULES:
        raise ValueError(f"无效数据集：{args.dataset}，可选值为 {sorted(SUPPORTED_MODULES.keys())}")

    recipe = build_recipe(args.method_name)
    set_seed(args.seed)

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

        _print_run_header(args, run_dir)
        recipe.validate_and_note(args)

        ctx = build_common_context(
            args=args,
            run_dir=run_dir,
            require_image_feature_cache=recipe.require_image_feature_cache,
        )

        state = recipe.build_state(ctx, args)

        config = {
            "method_name": args.method_name,
            **ctx.common_config,
            **recipe.build_config_extra(state, ctx, args),
        }
        save_json(run_dir / "config.json", config)

        print("[train] 开始训练 ...")
        best_val_loss = float("inf")
        best_state = None
        metrics_history = []

        for epoch in range(1, args.epochs + 1):
            train_row = recipe.train_one_epoch(state, ctx, args, epoch)
            val_metrics = recipe.evaluate_split(state, ctx.val_loader, ctx, args)

            row = {
                "epoch": epoch,
                **train_row,
                "val": val_metrics,
            }
            metrics_history.append(row)

            print(recipe.format_epoch_log(row, ctx, args))

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = recipe.build_best_state(
                    state=state,
                    ctx=ctx,
                    args=args,
                    epoch=epoch,
                    val_metrics=val_metrics,
                )
                torch.save(best_state, run_dir / recipe.best_checkpoint_filename)

            save_json(run_dir / "metrics_history.json", metrics_history)
            save_csv(run_dir / "metrics_history.csv", flatten_metrics_history(metrics_history))

        if best_state is None:
            raise RuntimeError("训练未产生 best checkpoint")

        best_ckpt_path = run_dir / recipe.best_checkpoint_filename
        if not best_ckpt_path.exists():
            raise RuntimeError(f"未找到最优权重文件：{best_ckpt_path}")

        print("[final] 加载最优权重并导出最终预测 ...")
        best_state = torch.load(best_ckpt_path, map_location=args.device)
        recipe.load_best_state(state, best_state, ctx, args)

        final_train_metrics = recipe.evaluate_split(state, ctx.train_eval_loader, ctx, args)
        final_val_metrics = recipe.evaluate_split(state, ctx.val_loader, ctx, args)
        final_test_metrics = recipe.evaluate_split(state, ctx.test_loader, ctx, args)

        recipe.dump_predictions(state, ctx, args)

        summary = {
            "method_name": args.method_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "best_epoch": best_state["best_epoch"],
            "best_val_metrics_saved": best_state["best_val_metrics"],
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


def run_text_only_bayes_coop(args) -> None:
    args.method_name = "text_only_bayes_coop"
    run_recipe_from_args(args)


def run_vlm_adapter(args) -> None:
    args.method_name = "vlm_adapter"
    run_recipe_from_args(args)