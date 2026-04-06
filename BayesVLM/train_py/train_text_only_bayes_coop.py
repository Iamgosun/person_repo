from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.data.dataset_ops import print_class_counts
from bayesvlm.data.factory import SUPPORTED_MODULES
from bayesvlm.data.pipeline import prepare_experiment_data
from bayesvlm.hessians import (
    load_hessians,
    optimize_prior_precision,
)
from bayesvlm.methods.text_only_bayes_coop import (
    build_text_only_bayes_coop_model,
    compute_text_covariance,
    dump_text_only_bayes_coop_predictions,
    evaluate_text_only_bayes_coop,
)
from bayesvlm.training.history import flatten_metrics_history
from bayesvlm.training.io import save_csv, save_json, tee_output
from bayesvlm.utils import (
    get_image_size,
    get_model_type_and_size,
    get_transform,
    load_model,
)
from bayesvlm.training.runtime import ensure_run_dir, set_seed



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
    shots_per_class: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 32,
    num_workers: int = 4,
    use_full_cov: bool = False,
    save_dir: str = "output",
    method_name: str = "text_only_bayes_coop",
    seed: int = 42,
    device: str = "cuda",
    prediction_topk: int = 5,
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

        print(f"[run] method={method_name}")
        print(f"[run] dataset={dataset}")
        print(f"[run] shots_per_class={shots_per_class}")
        print(f"[run] seed={seed}")
        print(f"[run] device={device}")
        print(f"[run] run_dir={run_dir}")

        if model_str not in MODEL_NAME_MAP:
            raise ValueError(f"无效模型名：{model_str}，可选值为 {list(MODEL_NAME_MAP.keys())}")

        if dataset not in SUPPORTED_MODULES:
            raise ValueError(f"无效数据集：{dataset}，可选值为 {sorted(SUPPORTED_MODULES.keys())}")

        model_type, _ = get_model_type_and_size(model_str)
        transform_image_size = get_image_size(model_str)
        transform = get_transform(model_type, transform_image_size)

        data = prepare_experiment_data(
            dataset=dataset,
            data_root=data_root,
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

        print("[1] 加载 Hessian 并优化文本投影层先验精度 ...")
        A_txt, B_txt = load_hessians(hessian_dir, tag="txt", return_info=False)

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
            use_full_cov=use_full_cov,
            device=device,
        )

        optimizer = torch.optim.AdamW(
            prompt_learner.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        config = {
            "method_name": method_name,
            "dataset": dataset,
            "hessian_dir": hessian_dir,
            "model_str": model_str,
            "local_model_path": local_model_path,
            "data_root": data_root,
            "pseudo_data_count": pseudo_data_count,
            "lambda_txt_init": lambda_txt_init,
            "lambda_opt_steps": lambda_opt_steps,
            "lambda_txt": lambda_txt,
            "n_ctx": n_ctx,
            "ctx_init": ctx_init,
            "shots_per_class": shots_per_class,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "use_full_cov": use_full_cov,
            "seed": seed,
            "device": device,
            "num_classes": len(data.class_names),
            "prediction_topk": prediction_topk,
            "run_dir": str(run_dir),
        }
        save_json(run_dir / "config.json", config)

        print("[2] 开始训练 CoOp 上下文 ...")
        best_val_loss = float("inf")
        best_state = None
        metrics_history = []

        for epoch in range(1, epochs + 1):
            model.train()
            prompt_learner.train()

            epoch_loss_sum = 0.0
            epoch_count = 0

            for batch in data.train_loader:
                labels = batch["class_id"].to(device)

                optimizer.zero_grad()
                prob_logits = model(batch=batch)
                loss = prob_logits.cross_entropy(
                    labels,
                    num_samples=0,
                    reduction="mean",
                )
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item() * labels.size(0)
                epoch_count += labels.size(0)

            train_metrics = evaluate_text_only_bayes_coop(
                model=model,
                loader=data.train_eval_loader,
                num_classes=len(data.class_names),
                device=device,
            )
            val_metrics = evaluate_text_only_bayes_coop(
                model=model,
                loader=data.val_loader,
                num_classes=len(data.class_names),
                device=device,
            )
            test_metrics = evaluate_text_only_bayes_coop(
                model=model,
                loader=data.test_loader,
                num_classes=len(data.class_names),
                device=device,
            )

            row = {
                "epoch": epoch,
                "train_loss_step_mean": epoch_loss_sum / max(epoch_count, 1),
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            }
            metrics_history.append(row)

            print(
                f"[Epoch {epoch:03d}] "
                f"train_acc={train_metrics['acc']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} "
                f"test_acc={test_metrics['acc']:.4f} "
                f"val_nlpd={val_metrics['nlpd']:.4f} "
                f"val_ece={val_metrics['ece']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {
                    "prompt_learner": prompt_learner.state_dict(),
                    "config": config,
                    "best_epoch": epoch,
                    "best_val_metrics": val_metrics,
                    "best_test_metrics": test_metrics,
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
            loader=data.train_eval_loader,
            num_classes=len(data.class_names),
            device=device,
        )
        final_val_metrics = evaluate_text_only_bayes_coop(
            model=model,
            loader=data.val_loader,
            num_classes=len(data.class_names),
            device=device,
        )
        final_test_metrics = evaluate_text_only_bayes_coop(
            model=model,
            loader=data.test_loader,
            num_classes=len(data.class_names),
            device=device,
        )

        dump_text_only_bayes_coop_predictions(
            run_dir=run_dir,
            split_name="train",
            model=model,
            loader=data.train_eval_loader,
            class_names=data.class_names,
            device=device,
            topk=prediction_topk,
        )
        dump_text_only_bayes_coop_predictions(
            run_dir=run_dir,
            split_name="val",
            model=model,
            loader=data.val_loader,
            class_names=data.class_names,
            device=device,
            topk=prediction_topk,
        )
        dump_text_only_bayes_coop_predictions(
            run_dir=run_dir,
            split_name="test",
            model=model,
            loader=data.test_loader,
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
            "best_test_metrics_saved": best_state["best_test_metrics"],
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
    parser.add_argument("--hessian_dir", type=str, default="hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K")
    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--local_model_path", type=str, default="./models/clip-vit-b32")
    parser.add_argument("--data_root", type=str, default="./datasets")

    parser.add_argument("--pseudo_data_count", type=int, default=4)
    parser.add_argument("--lambda_txt_init", type=float, default=300.0)
    parser.add_argument("--lambda_opt_steps", type=int, default=500)

    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--ctx_init", type=str, default="a photo of a")
    parser.add_argument("--shots_per_class", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_full_cov", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--method_name", type=str, default="text_only_bayes_coop")
    parser.add_argument("--prediction_topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

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
        shots_per_class=args.shots_per_class,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_full_cov=args.use_full_cov,
        save_dir=args.save_dir,
        method_name=args.method_name,
        prediction_topk=args.prediction_topk,
        seed=args.seed,
        device=args.device,
    )