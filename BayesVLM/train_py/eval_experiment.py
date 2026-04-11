from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesvlm.common import ProbabilisticLogits
from bayesvlm.data.pipeline import prepare_experiment_data
from bayesvlm.features.feature_dataset import build_feature_loader
from bayesvlm.features.image_cache import ImageFeatureCacheSpec, get_or_build_image_feature_bundle
from bayesvlm.methods.deterministic_coop import (
    _prepare_deterministic_eval_cache,
    collect_deterministic_coop_predictions,
)
from bayesvlm.methods.text_only_bayes_coop import (
    _prepare_text_only_bayes_eval_cache,
    collect_text_only_bayes_coop_predictions,
)
from bayesvlm.methods.vlm_adapter import (
    collect_vlm_adapter_predictions,
    reduce_logits_for_inference,
)
from bayesvlm.training.io import save_csv, save_json, save_jsonl, tee_output
from bayesvlm.training.metrics import build_official_bayesadapter_calibration_bins
from bayesvlm.training.ood_metrics import compute_ood_metrics_from_id_scores
from bayesvlm.utils import get_transform
from train_py.common_experiment import build_common_context
from train_py.recipes import build_recipe
from train_py.train_experiment import _ensure_common_flags


def _load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config/config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)



def _build_args_from_saved_config(cfg: dict) -> SimpleNamespace:
    args = SimpleNamespace(**cfg)

    if not hasattr(args, "model"):
        args.model = cfg["model_str"]
    if not hasattr(args, "local_model_path"):
        args.local_model_path = cfg["local_model_path_raw"]
    if not hasattr(args, "data_root"):
        args.data_root = cfg["data_root_raw"]
    if not hasattr(args, "save_dir"):
        args.save_dir = str(Path(cfg["run_dir"]).parent.parent.parent)

    _ensure_common_flags(args)
    return args



def _resolve_selected_checkpoint(run_dir: Path, recipe, model_selection: str) -> tuple[str, Path]:
    selection = str(model_selection).strip().lower()
    if selection == "best":
        return "best", run_dir / "checkpoints" / recipe.best_checkpoint_filename
    if selection == "last":
        return "last", run_dir / "checkpoints" / recipe.last_checkpoint_filename
    raise ValueError(f"未知 model_selection: {model_selection}")



def _normalize_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []



def _save_split_metrics_and_bins(
    split_dir: Path,
    metrics: dict[str, float],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    save_json(split_dir / "metrics.json", metrics)

    pred_path = split_dir / "predictions.pt"
    if not pred_path.exists():
        raise FileNotFoundError(f"找不到 predictions payload: {pred_path}")

    payload = torch.load(pred_path, map_location="cpu")
    if "probs" not in payload or "labels" not in payload:
        raise KeyError(
            f"{pred_path} 缺少 probs/labels，无法生成 calibration bins。"
            f"现有 keys={list(payload.keys())}"
        )

    rows = build_official_bayesadapter_calibration_bins(
        prediction=payload["probs"],
        label=payload["labels"],
        n_bins=10,
    )
    save_csv(split_dir / "calibration_bins.csv", rows)



def _build_external_test_loader(saved_args, base_ctx, target_dataset: str):
    eval_transform = get_transform(base_ctx.model_type, base_ctx.transform_image_size)

    data = prepare_experiment_data(
        dataset=target_dataset,
        data_root=str(base_ctx.data_root_path),
        batch_size=saved_args.batch_size,
        num_workers=saved_args.num_workers,
        train_transform=eval_transform,
        test_transform=eval_transform,
        shots_per_class=0,
        seed=saved_args.seed,
        shuffle_train=False,
        run_checks=False,
        run_loader_probe=False,
    )

    if saved_args.cache_image_features:
        test_features = get_or_build_image_feature_bundle(
            image_encoder=base_ctx.image_encoder,
            loader=data.test_loader,
            ds=data.test_ds,
            cache_root=base_ctx.image_feature_cache_root_path,
            spec=ImageFeatureCacheSpec(
                dataset=target_dataset,
                split="test",
                model_str=saved_args.model,
                local_model_path=base_ctx.local_model_path_cache_key,
                image_size=base_ctx.transform_image_size,
                transform_name=base_ctx.transform_name,
                data_root=str(base_ctx.data_root_path),
            ),
            force_rebuild=saved_args.rebuild_image_feature_cache,
        )
        test_loader = build_feature_loader(
            test_features,
            batch_size=saved_args.batch_size,
            num_workers=saved_args.num_workers,
            shuffle=False,
        )
    else:
        test_loader = data.test_loader

    return data.class_names, test_loader



def _collect_generalization_predictions(recipe_name: str, state, loader, class_names, device: str, split_name: str, topk: int):
    recipe_key = str(recipe_name).strip().lower()

    if recipe_key == "deterministic_coop":
        return collect_deterministic_coop_predictions(
            model=state["model"],
            loader=loader,
            class_names=class_names,
            device=device,
            split_name=split_name,
            topk=topk,
        )

    if recipe_key == "text_only_bayes_coop":
        return collect_text_only_bayes_coop_predictions(
            model=state["model"],
            loader=loader,
            class_names=class_names,
            device=device,
            split_name=split_name,
            topk=topk,
        )

    if recipe_key == "vlm_adapter":
        return collect_vlm_adapter_predictions(
            model=state["model"],
            loader=loader,
            class_names=class_names,
            device=device,
            split_name=split_name,
            topk=topk,
        )

    raise ValueError(f"暂不支持 recipe={recipe_name} 的泛化预测收集")



def _collect_ood_payload(recipe_name: str, state, loader, device: str) -> dict[str, torch.Tensor]:
    recipe_key = str(recipe_name).strip().lower()

    if recipe_key == "vlm_adapter":
        model = state["model"]
        model.eval()

        all_labels = []
        all_preds = []
        all_probs = []
        all_logits = []

        with torch.no_grad():
            for batch in loader:
                labels = batch["class_id"].to(device)
                raw_logits = model(batch=batch)
                logits = reduce_logits_for_inference(raw_logits)
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=1)

                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_probs.append(probs.detach().cpu())
                all_logits.append(logits.detach().cpu())

        return {
            "labels": torch.cat(all_labels, dim=0),
            "preds": torch.cat(all_preds, dim=0),
            "probs": torch.cat(all_probs, dim=0),
            "logits": torch.cat(all_logits, dim=0),
        }

    if recipe_key == "deterministic_coop":
        model = state["model"]
        model.eval()
        cache = _prepare_deterministic_eval_cache(model)
        text_features = cache["text_features"]

        all_labels = []
        all_preds = []
        all_probs = []
        all_logits = []

        with torch.no_grad():
            for batch in loader:
                labels = batch["class_id"].to(device)
                g = model.encode_image_batch(batch=batch)
                logits = model.vlm(g, text_features)
                probs = F.softmax(logits, dim=-1)
                preds = probs.argmax(dim=1)

                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_probs.append(probs.detach().cpu())
                all_logits.append(logits.detach().cpu())

        return {
            "labels": torch.cat(all_labels, dim=0),
            "preds": torch.cat(all_preds, dim=0),
            "probs": torch.cat(all_probs, dim=0),
            "logits": torch.cat(all_logits, dim=0),
        }

    if recipe_key == "text_only_bayes_coop":
        model = state["model"]
        model.eval()
        cache = _prepare_text_only_bayes_eval_cache(model)

        all_labels = []
        all_preds = []
        all_probs = []
        all_logits_mean = []
        all_logits_var = []

        with torch.no_grad():
            for batch in loader:
                labels = batch["class_id"].to(device)
                g = model.encode_image_batch(batch=batch).float()

                g_norm2 = (g ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
                g_norm = torch.sqrt(g_norm2)

                mean_cos = (g @ cache["mu_t"]) / (g_norm * cache["denom_text"].unsqueeze(0))

                if cache["use_full_cov"]:
                    g_quad = torch.einsum("bi,ij,bj->b", g, cache["B_inv"], g).unsqueeze(-1)
                else:
                    g_quad = ((g ** 2) * cache["diag_B"].unsqueeze(0)).sum(dim=-1, keepdim=True)

                denom_var = g_norm2 * (cache["mu_norm2"] + cache["trace_sigma"]).unsqueeze(0) + 1e-6
                var_cos = (g_quad * cache["alpha"].unsqueeze(0)) / denom_var
                var_cos = var_cos.clamp_min(0.0)

                logits_mean = mean_cos * cache["scale"]
                logits_var = var_cos * (cache["scale"] ** 2)

                if cache["logit_bias"] is not None:
                    logits_mean = logits_mean + cache["logit_bias"]

                prob_logits = ProbabilisticLogits(mean=logits_mean, var=logits_var)
                probs = prob_logits.softmax(num_samples=0)
                preds = probs.argmax(dim=1)

                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_probs.append(probs.detach().cpu())
                all_logits_mean.append(prob_logits.mean.detach().cpu())
                all_logits_var.append(prob_logits.var.detach().cpu())

        return {
            "labels": torch.cat(all_labels, dim=0),
            "preds": torch.cat(all_preds, dim=0),
            "probs": torch.cat(all_probs, dim=0),
            "logits_mean": torch.cat(all_logits_mean, dim=0),
            "logits_var": torch.cat(all_logits_var, dim=0),
        }

    raise ValueError(f"暂不支持 recipe={recipe_name} 的 OOD payload 收集")



def _compute_id_score_from_payload(payload: dict, score_name: str) -> torch.Tensor:
    score_key = str(score_name).strip().lower()

    if "probs" not in payload:
        raise KeyError(f"payload 缺少 probs，无法计算 OOD 分数。keys={list(payload.keys())}")

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
            raise KeyError(f"payload 缺少 logits/logits_mean，无法计算 energy。keys={list(payload.keys())}")
        return torch.logsumexp(logits, dim=1)

    raise ValueError(f"未知 OOD 分数: {score_name}")



def _run_ood_evaluation(run_dir: Path, saved_args, base_ctx, state) -> dict:
    target_datasets = _normalize_str_list(getattr(saved_args, "ood_datasets", []))
    if not target_datasets:
        return {"enabled": False, "targets": []}

    score_names = _normalize_str_list(getattr(saved_args, "ood_scores", ["msp", "entropy", "energy"]))
    if not score_names:
        score_names = ["msp", "entropy", "energy"]

    id_payload_path = run_dir / "eval" / "id" / "test" / "predictions.pt"
    if not id_payload_path.exists():
        raise FileNotFoundError(f"ID test predictions 不存在，无法进行 OOD 评估: {id_payload_path}")
    id_payload = torch.load(id_payload_path, map_location="cpu")

    summary_targets = []

    for target_dataset in target_datasets:
        if target_dataset == saved_args.dataset:
            continue

        print(f"[ood] target_dataset={target_dataset}")
        _, target_loader = _build_external_test_loader(saved_args, base_ctx, target_dataset)
        ood_payload = _collect_ood_payload(saved_args.recipe_name, state, target_loader, saved_args.device)

        dataset_entry = {
            "dataset": target_dataset,
            "num_id_samples": int(id_payload["probs"].shape[0]),
            "num_ood_samples": int(ood_payload["probs"].shape[0]),
            "scores": [],
        }

        for score_name in score_names:
            id_scores = _compute_id_score_from_payload(id_payload, score_name)
            ood_scores = _compute_id_score_from_payload(ood_payload, score_name)
            metrics = compute_ood_metrics_from_id_scores(id_scores, ood_scores)
            metrics.update(
                {
                    "score_name": score_name,
                    "score_semantics": "higher_means_more_ID_like",
                    "target_dataset": target_dataset,
                }
            )

            out_dir = run_dir / "eval" / "ood" / score_name / target_dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json(out_dir / "metrics.json", metrics)

            dataset_entry["scores"].append(
                {
                    "score_name": score_name,
                    "metrics_path": str((Path("eval") / "ood" / score_name / target_dataset / "metrics.json").as_posix()),
                }
            )

        summary_targets.append(dataset_entry)

    return {
        "enabled": True,
        "targets": summary_targets,
    }



def _run_generalization_evaluation(run_dir: Path, recipe, saved_args, base_ctx, state) -> dict:
    target_datasets = _normalize_str_list(getattr(saved_args, "generalization_datasets", []))
    if not target_datasets:
        return {"enabled": False, "targets": []}

    summary_targets = []

    for target_dataset in target_datasets:
        if target_dataset == saved_args.dataset:
            continue

        print(f"[generalization] target_dataset={target_dataset}")
        target_class_names, target_loader = _build_external_test_loader(saved_args, base_ctx, target_dataset)
        target_dir = run_dir / "eval" / "generalization" / target_dataset
        target_dir.mkdir(parents=True, exist_ok=True)

        if list(target_class_names) != list(base_ctx.class_names):
            skipped = {
                "enabled": False,
                "target_dataset": target_dataset,
                "skipped": True,
                "reason": "class_names mismatch with source run; generalization metrics only support identical label space.",
                "source_num_classes": len(base_ctx.class_names),
                "target_num_classes": len(target_class_names),
            }
            save_json(target_dir / "skipped.json", skipped)
            summary_targets.append(
                {
                    "dataset": target_dataset,
                    "skipped": True,
                    "reason": skipped["reason"],
                    "record_path": str((Path("eval") / "generalization" / target_dataset / "skipped.json").as_posix()),
                }
            )
            continue

        metrics = recipe.evaluate_split(state, target_loader, base_ctx, saved_args)
        rows, payload = _collect_generalization_predictions(
            recipe_name=saved_args.recipe_name,
            state=state,
            loader=target_loader,
            class_names=base_ctx.class_names,
            device=saved_args.device,
            split_name=f"generalization:{target_dataset}",
            topk=saved_args.prediction_topk,
        )

        save_jsonl(target_dir / "predictions.jsonl", rows)
        torch.save(payload, target_dir / "predictions.pt")
        _save_split_metrics_and_bins(target_dir, metrics)

        summary_targets.append(
            {
                "dataset": target_dataset,
                "skipped": False,
                "metrics_path": str((Path("eval") / "generalization" / target_dataset / "metrics.json").as_posix()),
                "predictions_pt": str((Path("eval") / "generalization" / target_dataset / "predictions.pt").as_posix()),
                "predictions_jsonl": str((Path("eval") / "generalization" / target_dataset / "predictions.jsonl").as_posix()),
                "bins_path": str((Path("eval") / "generalization" / target_dataset / "calibration_bins.csv").as_posix()),
            }
        )

    return {
        "enabled": True,
        "targets": summary_targets,
    }



def eval_recipe_from_args(args) -> None:
    _ensure_common_flags(args)

    recipe = build_recipe(args.recipe_name)

    run_dir = Path(args.save_dir) / args.method_name / args.dataset
    for part in recipe.run_path_parts(args):
        run_dir = run_dir / str(part)
    run_dir = run_dir / f"seed_{args.seed}"
    run_dir = run_dir.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"找不到 run_dir: {run_dir}")

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    with tee_output(eval_dir / "eval.log"):
        eval_start_time = time.time()

        cfg = _load_run_config(run_dir)
        saved_args = _build_args_from_saved_config(cfg)

        print(f"[eval] run_dir={run_dir}")
        print(f"[eval] recipe={saved_args.recipe_name}")
        print(f"[eval] method={saved_args.method_name}")
        print(f"[eval] dataset={saved_args.dataset}")
        print(f"[eval] model_selection={saved_args.model_selection}")

        ctx = build_common_context(
            args=saved_args,
            run_dir=run_dir,
            require_image_feature_cache=recipe.require_image_feature_cache,
        )
        state = recipe.build_state(ctx, saved_args)

        selected_label, selected_ckpt_path = _resolve_selected_checkpoint(
            run_dir=run_dir,
            recipe=recipe,
            model_selection=saved_args.model_selection,
        )
        if not selected_ckpt_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {selected_ckpt_path}")

        print(f"[eval] 加载 {selected_label} checkpoint: {selected_ckpt_path}")
        selected_state = torch.load(selected_ckpt_path, map_location=saved_args.device)
        recipe.load_best_state(state, selected_state, ctx, saved_args)

        final_train_metrics = recipe.evaluate_split(state, ctx.train_eval_loader, ctx, saved_args)
        final_val_metrics = recipe.evaluate_split(state, ctx.val_loader, ctx, saved_args)
        final_test_metrics = recipe.evaluate_split(state, ctx.test_loader, ctx, saved_args)

        recipe.dump_predictions(state, ctx, saved_args)

        _save_split_metrics_and_bins(run_dir / "eval" / "id" / "train", final_train_metrics)
        _save_split_metrics_and_bins(run_dir / "eval" / "id" / "val", final_val_metrics)
        _save_split_metrics_and_bins(run_dir / "eval" / "id" / "test", final_test_metrics)

        ood_summary = _run_ood_evaluation(run_dir, saved_args, ctx, state)
        generalization_summary = _run_generalization_evaluation(run_dir, recipe, saved_args, ctx, state)

        summary = {
            "recipe_name": saved_args.recipe_name,
            "method_name": saved_args.method_name,
            "dataset": saved_args.dataset,
            "seed": saved_args.seed,
            "model_selection": saved_args.model_selection,
            "selection_metric": cfg.get("selection_metric"),
            "selection_mode": cfg.get("selection_mode"),
            "selected_checkpoint": selected_label,
            "selected_epoch": selected_state.get("_checkpoint_epoch"),
            "selected_val_metrics": selected_state.get("_checkpoint_val_metrics"),
            "final_train_metrics_recomputed": final_train_metrics,
            "final_val_metrics_recomputed": final_val_metrics,
            "final_test_metrics_recomputed": final_test_metrics,
            "ood_evaluation": ood_summary,
            "generalization_evaluation": generalization_summary,
            "run_dir": str(run_dir),
            "elapsed_seconds": round(time.time() - eval_start_time, 2),
            "artifacts": {
                "config_file": "config/config.json",
                "class_names_file": "config/class_names.json",
                "best_ckpt_file": f"checkpoints/{recipe.best_checkpoint_filename}",
                "last_ckpt_file": f"checkpoints/{recipe.last_checkpoint_filename}",
                "eval_log_file": "eval/eval.log",
                "id_train_metrics": "eval/id/train/metrics.json",
                "id_train_bins": "eval/id/train/calibration_bins.csv",
                "id_train_predictions_pt": "eval/id/train/predictions.pt",
                "id_train_predictions_jsonl": "eval/id/train/predictions.jsonl",
                "id_val_metrics": "eval/id/val/metrics.json",
                "id_val_bins": "eval/id/val/calibration_bins.csv",
                "id_val_predictions_pt": "eval/id/val/predictions.pt",
                "id_val_predictions_jsonl": "eval/id/val/predictions.jsonl",
                "id_test_metrics": "eval/id/test/metrics.json",
                "id_test_bins": "eval/id/test/calibration_bins.csv",
                "id_test_predictions_pt": "eval/id/test/predictions.pt",
                "id_test_predictions_jsonl": "eval/id/test/predictions.jsonl",
            },
        }
        summary.update(recipe.build_summary_extra(state, selected_state, ctx, saved_args))
        save_json(run_dir / "summary.json", summary)

        print("[done] 评估阶段完成：")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
