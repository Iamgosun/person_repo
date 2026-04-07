from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.data.dataset_ops import print_class_counts, unwrap_dataset_and_indices
from bayesvlm.data.factory import SUPPORTED_MODULES
from bayesvlm.data.pipeline import build_loader, prepare_experiment_data
from bayesvlm.features.feature_dataset import build_feature_loader
from bayesvlm.features.image_cache import (
    ImageFeatureCacheSpec,
    get_or_build_image_feature_bundle,
)
from bayesvlm.hessians import load_hessians, optimize_prior_precision
from bayesvlm.methods.text_only_bayes_coop import (
    build_text_only_bayes_coop_model,
    compute_text_covariance,
    compute_text_only_bayes_coop_train_losses,
    dump_text_only_bayes_coop_predictions,
    evaluate_text_only_bayes_coop,
)
from bayesvlm.training.history import flatten_metrics_history
from bayesvlm.training.io import save_csv, save_json, tee_output
from bayesvlm.training.runtime import ensure_run_dir, set_seed
from bayesvlm.utils import (
    get_image_size,
    get_model_type_and_size,
    get_transform,
    load_model,
)


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p.resolve()

    p2 = (Path.cwd() / path_str).resolve()
    if p2.exists():
        return p2

    raise FileNotFoundError(f"路径不存在：{path_str}")


def _normalize_path_for_cache(path_str: str | None) -> str | None:
    if path_str is None:
        return None

    p = Path(path_str)
    if p.exists():
        return str(p.resolve())

    p2 = (Path.cwd() / path_str).resolve()
    if p2.exists():
        return str(p2)

    return path_str


def _stable_transform_name(model_type: str, image_size: int) -> str:
    if model_type == "siglip":
        return f"siglip_transform(image_size={image_size})"
    return f"default_transform(image_size={image_size})"


def _check_txt_hessian_dir(hessian_dir: Path) -> dict:
    required = [
        hessian_dir / "A_txt_analytic.pt",
        hessian_dir / "B_txt_analytic.pt",
    ]
    missing = [str(p) for p in required if not p.exists()]

    return {
        "ok": len(missing) == 0,
        "missing_required": missing,
        "dir": str(hessian_dir),
    }


def _compose_train_loss(
    *,
    objective: str,
    epoch: int,
    hybrid_warmup_epochs: int,
    loss_dict: dict[str, torch.Tensor],
    map_loss_weight: float,
    bayes_loss_weight: float,
    ctx_reg_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    map_loss = loss_dict["map_loss"]
    bayes_loss = loss_dict["bayes_loss"]
    ctx_reg = loss_dict["ctx_reg"]

    if objective == "map":
        loss = map_loss + ctx_reg_weight * ctx_reg
    elif objective == "bayes":
        loss = bayes_loss + ctx_reg_weight * ctx_reg
    elif objective == "hybrid":
        if epoch <= hybrid_warmup_epochs:
            loss = map_loss + ctx_reg_weight * ctx_reg
        else:
            loss = (
                map_loss_weight * map_loss
                + bayes_loss_weight * bayes_loss
                + ctx_reg_weight * ctx_reg
            )
    else:
        raise ValueError(f"未知 train_objective: {objective}")

    stats = {
        "map_loss": float(map_loss.detach().item()),
        "bayes_loss": float(bayes_loss.detach().item()),
        "ctx_reg": float(ctx_reg.detach().item()),
        "total_loss": float(loss.detach().item()),
    }
    return loss, stats


def main(
    dataset: str,
    hessian_dir: str,
    model_str: str = "clip-base",
    local_model_path: str = "./models/clip-vit-b32",
    data_root: str = "./datasets",
    pseudo_data_count: int = 4,
    lambda_txt_init: float = 300.0,
    lambda_opt_steps: int = 500,
    n_ctx: int = 16,
    ctx_init: str = "a photo of a",
    csc: bool = False,
    class_token_position: str = "end",
    shots_per_class: int = 16,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    epochs: int = 50,
    batch_size: int = 32,
    num_workers: int = 4,
    use_full_cov: bool = False,
    train_objective: str = "hybrid",
    hybrid_warmup_epochs: int = 5,
    map_loss_weight: float = 1.0,
    bayes_loss_weight: float = 1.0,
    ctx_reg_weight: float = 1e-4,
    save_dir: str = "output",
    method_name: str = "text_only_bayes_coop",
    seed: int = 42,
    device: str = "cuda",
    prediction_topk: int = 5,
    cache_image_features: bool = True,
    image_feature_cache_root: str = "./cache/image_features",
    rebuild_image_feature_cache: bool = False,
) -> None:
    set_seed(seed)

    run_dir = ensure_run_dir(
        save_dir=save_dir,
        method_name=method_name,
        dataset=dataset,
        seed=seed,
        path_parts=[f"shot_{shots_per_class}"],
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_output(run_dir / "train.log"):
        run_start_time = time.time()

        hessian_dir_path = _resolve_path(hessian_dir)
        data_root_path = _resolve_path(data_root)
        image_feature_cache_root_path = Path(image_feature_cache_root).resolve()
        local_model_path_for_cache = _normalize_path_for_cache(local_model_path)

        print(f"[run] method={method_name}")
        print(f"[run] dataset={dataset}")
        print(f"[run] shots_per_class={shots_per_class}")
        print(f"[run] seed={seed}")
        print(f"[run] device={device}")
        print(f"[run] run_dir={run_dir}")
        print(f"[run] train_objective={train_objective}")
        print(f"[run] hybrid_warmup_epochs={hybrid_warmup_epochs}")
        print(f"[run] map_loss_weight={map_loss_weight}")
        print(f"[run] bayes_loss_weight={bayes_loss_weight}")
        print(f"[run] ctx_reg_weight={ctx_reg_weight}")
        print(f"[run] cache_image_features={cache_image_features}")
        print(f"[run] image_feature_cache_root={image_feature_cache_root_path}")
        print(f"[run] rebuild_image_feature_cache={rebuild_image_feature_cache}")
        print(f"[run] local_model_path(raw)={local_model_path}")
        print(f"[run] local_model_path(cache_key)={local_model_path_for_cache}")
        print(f"[run] data_root(raw)={data_root}")
        print(f"[run] data_root(cache_key)={data_root_path}")

        if model_str not in MODEL_NAME_MAP:
            raise ValueError(f"无效模型名：{model_str}，可选值为 {list(MODEL_NAME_MAP.keys())}")

        if dataset not in SUPPORTED_MODULES:
            raise ValueError(f"无效数据集：{dataset}，可选值为 {sorted(SUPPORTED_MODULES.keys())}")

        hessian_check = _check_txt_hessian_dir(hessian_dir_path)
        if not hessian_check["ok"]:
            existing_files = []
            if hessian_dir_path.exists():
                existing_files = sorted([p.name for p in hessian_dir_path.iterdir()])

            raise FileNotFoundError(
                "当前 text_only_bayes_coop 训练只会读取 txt Hessian。\n"
                f"hessian_dir = {hessian_dir_path}\n"
                f"缺少文件: {hessian_check['missing_required']}\n"
                f"目录现有文件: {existing_files}\n\n"
                "该方法至少需要：\n"
                "  - A_txt_analytic.pt\n"
                "  - B_txt_analytic.pt\n"
            )

        model_type, _ = get_model_type_and_size(model_str)
        transform_image_size = get_image_size(model_str)
        transform = get_transform(model_type, transform_image_size)
        transform_name = _stable_transform_name(model_type, transform_image_size)

        print(f"[run] model_type={model_type}")
        print(f"[run] image_size={transform_image_size}")
        print(f"[run] transform_name(cache_key)={transform_name}")

        data = prepare_experiment_data(
            dataset=dataset,
            data_root=str(data_root_path),
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=transform,
            test_transform=transform,
            shots_per_class=shots_per_class,
            seed=seed,
            shuffle_train=True,
            run_checks=False,
            run_loader_probe=False,
        )

        print_class_counts(data.train_ds, split_name="train")
        print_class_counts(data.test_ds, split_name="test")
        save_json(run_dir / "class_names.json", {"class_names": data.class_names})

        image_encoder, text_encoder, vlm = load_model(
            model_str=model_str,
            device=device,
            local_model_path=local_model_path,
        )

        if cache_image_features:
            print("[0] 构建/加载共享图像特征缓存 ...")

            raw_train_loader_for_cache = build_loader(
                data.raw_train_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )

            train_full_features = get_or_build_image_feature_bundle(
                image_encoder=image_encoder,
                loader=raw_train_loader_for_cache,
                ds=data.raw_train_ds,
                cache_root=image_feature_cache_root_path,
                spec=ImageFeatureCacheSpec(
                    dataset=dataset,
                    split="train_full",
                    model_str=model_str,
                    local_model_path=local_model_path_for_cache,
                    image_size=transform_image_size,
                    transform_name=transform_name,
                    data_root=str(data_root_path),
                ),
                force_rebuild=rebuild_image_feature_cache,
            )

            val_features = get_or_build_image_feature_bundle(
                image_encoder=image_encoder,
                loader=data.val_loader,
                ds=data.val_ds,
                cache_root=image_feature_cache_root_path,
                spec=ImageFeatureCacheSpec(
                    dataset=dataset,
                    split="val",
                    model_str=model_str,
                    local_model_path=local_model_path_for_cache,
                    image_size=transform_image_size,
                    transform_name=transform_name,
                    data_root=str(data_root_path),
                ),
                force_rebuild=rebuild_image_feature_cache,
            )

            test_features = get_or_build_image_feature_bundle(
                image_encoder=image_encoder,
                loader=data.test_loader,
                ds=data.test_ds,
                cache_root=image_feature_cache_root_path,
                spec=ImageFeatureCacheSpec(
                    dataset=dataset,
                    split="test",
                    model_str=model_str,
                    local_model_path=local_model_path_for_cache,
                    image_size=transform_image_size,
                    transform_name=transform_name,
                    data_root=str(data_root_path),
                ),
                force_rebuild=rebuild_image_feature_cache,
            )

            _, train_indices = unwrap_dataset_and_indices(data.train_ds)
            if train_indices is None:
                train_indices = list(range(len(data.train_ds)))

            print(
                f"[cache] raw_train_samples={len(data.raw_train_ds)} | "
                f"fewshot_train_samples={len(train_indices)}"
            )

            train_subset_features = train_full_features.subset(train_indices)

            train_loader = build_feature_loader(
                train_subset_features,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )
            train_eval_loader = build_feature_loader(
                train_subset_features,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
            val_loader = build_feature_loader(
                val_features,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
            test_loader = build_feature_loader(
                test_features,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
        else:
            train_loader = data.train_loader
            train_eval_loader = data.train_eval_loader
            val_loader = data.val_loader
            test_loader = data.test_loader

        print("[1] 加载 txt Hessian 并优化文本投影层先验精度 ...")
        print(f"    hessian_dir = {hessian_dir_path}")

        A_txt, B_txt = load_hessians(str(hessian_dir_path), tag="txt", return_info=False)

        lambda_txt = optimize_prior_precision(
            projection=text_encoder.text_projection,
            A=A_txt,
            B=B_txt,
            lmbda_init=lambda_txt_init,
            n=pseudo_data_count,
            lr=1e-2,
            num_steps=lambda_opt_steps,
            device=device,
            verbose=True,
        ).item()

        print(f"    n_txt      = {pseudo_data_count}")
        print(f"    lambda_txt = {lambda_txt:.6f}")

        text_covariance = compute_text_covariance(
            A_txt=A_txt.to(device),
            B_txt=B_txt.to(device),
            n_txt=pseudo_data_count,
            lambda_txt=lambda_txt,
        )

        prompt_learner, model = build_text_only_bayes_coop_model(
            class_names=data.class_names,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            vlm=vlm,
            text_covariance=text_covariance,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            csc=csc,
            class_token_position=class_token_position,
            use_full_cov=use_full_cov,
            device=device,
        )

        ctx_anchor = prompt_learner.ctx.detach().clone()

        optimizer = torch.optim.SGD(
            prompt_learner.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

        config = {
            "method_name": method_name,
            "dataset": dataset,
            "hessian_dir": str(hessian_dir_path),
            "model_str": model_str,
            "local_model_path_raw": local_model_path,
            "local_model_path_cache_key": local_model_path_for_cache,
            "data_root_raw": data_root,
            "data_root_cache_key": str(data_root_path),
            "pseudo_data_count": pseudo_data_count,
            "lambda_txt_init": lambda_txt_init,
            "lambda_opt_steps": lambda_opt_steps,
            "lambda_txt": lambda_txt,
            "n_ctx": n_ctx,
            "ctx_init": ctx_init,
            "csc": csc,
            "class_token_position": class_token_position,
            "shots_per_class": shots_per_class,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "use_full_cov": use_full_cov,
            "train_objective": train_objective,
            "hybrid_warmup_epochs": hybrid_warmup_epochs,
            "map_loss_weight": map_loss_weight,
            "bayes_loss_weight": bayes_loss_weight,
            "ctx_reg_weight": ctx_reg_weight,
            "seed": seed,
            "device": device,
            "num_classes": len(data.class_names),
            "prediction_topk": prediction_topk,
            "run_dir": str(run_dir),
            "cache_image_features": cache_image_features,
            "image_feature_cache_root": str(image_feature_cache_root_path),
            "rebuild_image_feature_cache": rebuild_image_feature_cache,
            "transform_name_cache_key": transform_name,
            "hessian_check": hessian_check,
        }
        save_json(run_dir / "config.json", config)

        print("[2] 开始训练 Text-Only Bayes CoOp ...")
        best_val_acc = float("-inf")
        best_state = None
        metrics_history = []

        for epoch in range(1, epochs + 1):
            model.train()
            prompt_learner.train()

            epoch_total_loss = 0.0
            epoch_map_loss = 0.0
            epoch_bayes_loss = 0.0
            epoch_ctx_reg = 0.0
            epoch_count = 0

            for batch in train_loader:
                labels = batch["class_id"].to(device)

                optimizer.zero_grad()
                loss_dict = compute_text_only_bayes_coop_train_losses(
                    model=model,
                    prompt_learner=prompt_learner,
                    batch=batch,
                    labels=labels,
                    ctx_anchor=ctx_anchor,
                    ctx_reg_weight=ctx_reg_weight,
                )
                loss, stats = _compose_train_loss(
                    objective=train_objective,
                    epoch=epoch,
                    hybrid_warmup_epochs=hybrid_warmup_epochs,
                    loss_dict=loss_dict,
                    map_loss_weight=map_loss_weight,
                    bayes_loss_weight=bayes_loss_weight,
                    ctx_reg_weight=ctx_reg_weight,
                )
                loss.backward()
                optimizer.step()

                batch_n = labels.size(0)
                epoch_total_loss += stats["total_loss"] * batch_n
                epoch_map_loss += stats["map_loss"] * batch_n
                epoch_bayes_loss += stats["bayes_loss"] * batch_n
                epoch_ctx_reg += stats["ctx_reg"] * batch_n
                epoch_count += batch_n

            scheduler.step()

            val_metrics = evaluate_text_only_bayes_coop(
                model=model,
                loader=val_loader,
                num_classes=len(data.class_names),
                device=device,
            )

            row = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "train_loss_step_mean": epoch_total_loss / max(epoch_count, 1),
                "train_map_loss_mean": epoch_map_loss / max(epoch_count, 1),
                "train_bayes_loss_mean": epoch_bayes_loss / max(epoch_count, 1),
                "train_ctx_reg_mean": epoch_ctx_reg / max(epoch_count, 1),
                "val": val_metrics,
            }
            metrics_history.append(row)

            print(
                f"[Epoch {epoch:03d}] "
                f"lr={row['lr']:.6f} "
                f"train_total={row['train_loss_step_mean']:.4f} "
                f"train_map={row['train_map_loss_mean']:.4f} "
                f"train_bayes={row['train_bayes_loss_mean']:.4f} "
                f"train_ctx_reg={row['train_ctx_reg_mean']:.6f} "
                f"val_acc={val_metrics['acc']:.4f} "
                f"val_nlpd={val_metrics['nlpd']:.4f} "
                f"val_ece={val_metrics['ece']:.4f}"
            )

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_state = {
                    "prompt_learner": prompt_learner.state_dict(),
                    "config": config,
                    "best_epoch": epoch,
                    "best_val_metrics": val_metrics,
                }
                torch.save(best_state, run_dir / "best_prompt_learner.pt")

            save_json(run_dir / "metrics_history.json", metrics_history)
            save_csv(run_dir / "metrics_history.csv", flatten_metrics_history(metrics_history))

        print("[3] 训练结束，加载最优 prompt 并导出最终预测 ...")
        best_ckpt_path = run_dir / "best_prompt_learner.pt"
        if not best_ckpt_path.exists():
            raise RuntimeError(f"未找到最优权重文件：{best_ckpt_path}")

        best_state = torch.load(best_ckpt_path, map_location=device)
        prompt_learner.load_state_dict(best_state["prompt_learner"])

        final_train_metrics = evaluate_text_only_bayes_coop(
            model=model,
            loader=train_eval_loader,
            num_classes=len(data.class_names),
            device=device,
        )
        final_val_metrics = evaluate_text_only_bayes_coop(
            model=model,
            loader=val_loader,
            num_classes=len(data.class_names),
            device=device,
        )
        final_test_metrics = evaluate_text_only_bayes_coop(
            model=model,
            loader=test_loader,
            num_classes=len(data.class_names),
            device=device,
        )

        dump_text_only_bayes_coop_predictions(
            run_dir=run_dir,
            split_name="train",
            model=model,
            loader=train_eval_loader,
            class_names=data.class_names,
            device=device,
            topk=prediction_topk,
        )
        dump_text_only_bayes_coop_predictions(
            run_dir=run_dir,
            split_name="val",
            model=model,
            loader=val_loader,
            class_names=data.class_names,
            device=device,
            topk=prediction_topk,
        )
        dump_text_only_bayes_coop_predictions(
            run_dir=run_dir,
            split_name="test",
            model=model,
            loader=test_loader,
            class_names=data.class_names,
            device=device,
            topk=prediction_topk,
        )

        summary = {
            "method_name": method_name,
            "dataset": dataset,
            "seed": seed,
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
                "best_ckpt_file": "best_prompt_learner.pt",
                "train_predictions_jsonl": "train_predictions.jsonl",
                "train_predictions_pt": "train_predictions.pt",
                "val_predictions_jsonl": "val_predictions.jsonl",
                "val_predictions_pt": "val_predictions.pt",
                "test_predictions_jsonl": "test_predictions.jsonl",
                "test_predictions_pt": "test_predictions.pt",
            },
        }

        save_json(run_dir / "summary.json", summary)

        print("[done] 最终结果：")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--hessian_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--local_model_path", type=str, default="./models/clip-vit-b32")
    parser.add_argument("--data_root", type=str, default="./datasets")

    parser.add_argument("--pseudo_data_count", type=int, default=4)
    parser.add_argument("--lambda_txt_init", type=float, default=300.0)
    parser.add_argument("--lambda_opt_steps", type=int, default=500)

    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--ctx_init", type=str, default="a photo of a")
    parser.add_argument("--csc", action="store_true", default=False)
    parser.add_argument("--class_token_position", type=str, default="end")
    parser.add_argument("--shots_per_class", type=int, default=16)

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_full_cov", action="store_true", default=False)
    parser.add_argument("--train_objective", type=str, default="hybrid", choices=["map", "bayes", "hybrid"])
    parser.add_argument("--hybrid_warmup_epochs", type=int, default=5)
    parser.add_argument("--map_loss_weight", type=float, default=1.0)
    parser.add_argument("--bayes_loss_weight", type=float, default=1.0)
    parser.add_argument("--ctx_reg_weight", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--method_name", type=str, default="text_only_bayes_coop")
    parser.add_argument("--prediction_topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--disable_cache_image_features", action="store_true", default=False)
    parser.add_argument("--image_feature_cache_root", type=str, default="./cache/image_features")
    parser.add_argument("--rebuild_image_feature_cache", action="store_true", default=False)

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        hessian_dir=args.hessian_dir,
        model_str=args.model,
        local_model_path=args.local_model_path,
        data_root=args.data_root,
        pseudo_data_count=args.pseudo_data_count,
        lambda_txt_init=args.lambda_txt_init,
        lambda_opt_steps=args.lambda_opt_steps,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        csc=args.csc,
        class_token_position=args.class_token_position,
        shots_per_class=args.shots_per_class,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_full_cov=args.use_full_cov,
        train_objective=args.train_objective,
        hybrid_warmup_epochs=args.hybrid_warmup_epochs,
        map_loss_weight=args.map_loss_weight,
        bayes_loss_weight=args.bayes_loss_weight,
        ctx_reg_weight=args.ctx_reg_weight,
        save_dir=args.save_dir,
        method_name=args.method_name,
        prediction_topk=args.prediction_topk,
        seed=args.seed,
        device=args.device,
        cache_image_features=not args.disable_cache_image_features,
        image_feature_cache_root=args.image_feature_cache_root,
        rebuild_image_feature_cache=args.rebuild_image_feature_cache,
    )
