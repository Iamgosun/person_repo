from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.data.dataset_ops import print_class_counts
from bayesvlm.data.factory import SUPPORTED_MODULES
from bayesvlm.data.pipeline import prepare_experiment_data
from bayesvlm.methods.vlm_adapter import (
    build_vlm_adapter_model,
    compute_adapter_regularization_loss,
    compute_crossmodal_text_loss,
    dump_vlm_adapter_predictions,
    evaluate_vlm_adapter,
    evaluate_zero_shot_vlm_adapter,
    maybe_init_special_adapter,
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
    model_str: str = "clip-base",
    local_model_path: str = "./models/clip-vit-b32",
    data_root: str = "./datasets",
    adapter_name: str = "LP",
    initialization: str = "MEAN",
    shots_per_class: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 32,
    num_workers: int = 4,
    save_dir: str = "output",
    method_name: str = "vlm_adapter",
    seed: int = 42,
    device: str = "cuda",
    prediction_topk: int = 5,
    hessian_dir: str | None = None,
    pseudo_data_count: int = 4,
    taskres_alpha: float = 0.5,
    clipa_ratio: float = 0.2,
    clipa_hidden_dim: int = 0,
    tipa_alpha: float = 1.0,
    tipa_beta: float = 1.0,
    gaussian_prior_sigma: float = 0.01,
    gaussian_mc_samples: int = 3,
    gaussian_anneal_start_epoch: int = 20,
) -> None:
    set_seed(seed)

    run_dir = ensure_run_dir(
        save_dir=save_dir,
        method_name=method_name,
        dataset=dataset,
        seed=seed,
        path_parts=[
            adapter_name.upper(),
            initialization,
            f"shot_{shots_per_class}",
        ],
    )


    run_dir.mkdir(parents=True, exist_ok=True)

    with tee_output(run_dir / "train.log"):
        run_start_time = time.time()

        print(f"[run] method={method_name}")
        print(f"[run] dataset={dataset}")
        print(f"[run] adapter={adapter_name}")
        print(f"[run] initialization={initialization}")
        print(f"[run] shots_per_class={shots_per_class}")
        print(f"[run] seed={seed}")
        print(f"[run] device={device}")
        print(f"[run] run_dir={run_dir}")

        if hessian_dir:
            print(f"[note] hessian_dir={hessian_dir} 当前 raw-image adapter 训练不会直接使用。")
        print(f"[note] pseudo_data_count={pseudo_data_count} 仅为兼容旧接口保留。")

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

        cfg = {
            "model": model_str,
            "model_name_or_path": local_model_path,
            "datasetname": dataset,
            "adapter_name": adapter_name,
            "initialization": initialization,
            "device": device,
            "epochs": epochs,
            "taskres_alpha": taskres_alpha,
            "clipa_ratio": clipa_ratio,
            "clipa_hidden_dim": None if clipa_hidden_dim <= 0 else clipa_hidden_dim,
            "tipa_alpha": tipa_alpha,
            "tipa_beta": tipa_beta,
            "gaussian_prior_sigma": gaussian_prior_sigma,
            "gaussian_mc_samples": gaussian_mc_samples,
            "gaussian_anneal_start_epoch": gaussian_anneal_start_epoch,
        }

        model = build_vlm_adapter_model(
            cfg=cfg,
            class_names=data.class_names,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            vlm=vlm,
            device=device,
        )

        maybe_init_special_adapter(
            model=model,
            adapter_name=adapter_name,
            train_eval_loader=data.train_eval_loader,
            device=device,
        )

        zero_shot_test = evaluate_zero_shot_vlm_adapter(
            model=model,
            loader=data.test_loader,
            num_classes=len(data.class_names),
            device=device,
        )
        print(
            f"[zero-shot] "
            f"test_acc={zero_shot_test['acc']:.4f} "
            f"test_nlpd={zero_shot_test['nlpd']:.4f} "
            f"test_ece={zero_shot_test['ece']:.4f}"
        )

        optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        config = {
            "method_name": method_name,
            "dataset": dataset,
            "model_str": model_str,
            "local_model_path": local_model_path,
            "data_root": data_root,
            "adapter_name": adapter_name,
            "initialization": initialization,
            "shots_per_class": shots_per_class,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seed": seed,
            "device": device,
            "prediction_topk": prediction_topk,
            "hessian_dir_ignored": hessian_dir,
            "pseudo_data_count_ignored": pseudo_data_count,
            "taskres_alpha": taskres_alpha,
            "clipa_ratio": clipa_ratio,
            "clipa_hidden_dim": None if clipa_hidden_dim <= 0 else clipa_hidden_dim,
            "tipa_alpha": tipa_alpha,
            "tipa_beta": tipa_beta,
            "gaussian_prior_sigma": gaussian_prior_sigma,
            "gaussian_mc_samples": gaussian_mc_samples,
            "gaussian_anneal_start_epoch": gaussian_anneal_start_epoch,
            "zero_shot_test": zero_shot_test,
            "run_dir": str(run_dir),
        }
        save_json(run_dir / "config.json", config)

        best_val_loss = float("inf")
        best_state = None
        metrics_history = []

        print("[train] 开始 adapter 训练 ...")
        for epoch in range(1, epochs + 1):
            model.train()
            if hasattr(model, "set_epoch"):
                model.set_epoch(epoch)

            epoch_loss_sum = 0.0
            epoch_reg_sum = 0.0
            epoch_crossmodal_text_sum = 0.0
            epoch_count = 0

            for batch in data.train_loader:
                labels = batch["class_id"].to(device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(batch=batch)
                ce_loss = F.cross_entropy(logits, labels)

                reg_loss, reg_info = compute_adapter_regularization_loss(model)
                total_loss = ce_loss + reg_loss

                if adapter_name.upper() == "CROSSMODAL":
                    aux_text_loss = compute_crossmodal_text_loss(
                        model=model,
                        batch_size=labels.size(0),
                        device=device,
                    )
                    total_loss = total_loss + aux_text_loss
                    epoch_crossmodal_text_sum += aux_text_loss.item() * labels.size(0)

                total_loss.backward()
                optimizer.step()

                epoch_loss_sum += total_loss.item() * labels.size(0)
                epoch_reg_sum += reg_loss.item() * labels.size(0)
                epoch_count += labels.size(0)

            train_metrics = evaluate_vlm_adapter(
                model=model,
                loader=data.train_eval_loader,
                num_classes=len(data.class_names),
                device=device,
            )
            val_metrics = evaluate_vlm_adapter(
                model=model,
                loader=data.val_loader,
                num_classes=len(data.class_names),
                device=device,
            )
            test_metrics = evaluate_vlm_adapter(
                model=model,
                loader=data.test_loader,
                num_classes=len(data.class_names),
                device=device,
            )

            row = {
                "epoch": epoch,
                "train_loss_step_mean": epoch_loss_sum / max(epoch_count, 1),
                "loss_reg": epoch_reg_sum / max(epoch_count, 1),
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            }

            for key in ["loss_kl_raw", "loss_kl", "kl_weight"]:
                if key in reg_info:
                    row[key] = reg_info[key]

            if adapter_name.upper() == "CROSSMODAL":
                row["loss_crossmodal_text"] = epoch_crossmodal_text_sum / max(epoch_count, 1)

            metrics_history.append(row)

            log_msg = (
                f"[Epoch {epoch:03d}] "
                f"train_acc={train_metrics['acc']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} "
                f"test_acc={test_metrics['acc']:.4f} "
                f"val_nlpd={val_metrics['nlpd']:.4f} "
                f"val_ece={val_metrics['ece']:.4f}"
            )
            if "kl_weight" in row:
                log_msg += f" kl_weight={row['kl_weight']:.4f}"
            if "loss_kl_raw" in row:
                log_msg += f" loss_kl_raw={row['loss_kl_raw']:.4f}"
            print(log_msg)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {
                    "adapter": copy.deepcopy(model.adapter.state_dict()),
                    "best_epoch": epoch,
                    "best_val_metrics": val_metrics,
                    "best_test_metrics": test_metrics,
                    "config": config,
                }
                torch.save(best_state, run_dir / "best_adapter.pt")

            save_json(run_dir / "metrics_history.json", metrics_history)
            save_csv(run_dir / "metrics_history.csv", flatten_metrics_history(metrics_history))

        if best_state is None:
            raise RuntimeError("训练未产生 best checkpoint")

        print("[final] 加载最优 adapter 并导出最终预测 ...")
        model.adapter.load_state_dict(best_state["adapter"])

        final_train_metrics = evaluate_vlm_adapter(
            model=model,
            loader=data.train_eval_loader,
            num_classes=len(data.class_names),
            device=device,
        )
        final_val_metrics = evaluate_vlm_adapter(
            model=model,
            loader=data.val_loader,
            num_classes=len(data.class_names),
            device=device,
        )
        final_test_metrics = evaluate_vlm_adapter(
            model=model,
            loader=data.test_loader,
            num_classes=len(data.class_names),
            device=device,
        )

        dump_vlm_adapter_predictions(
            run_dir=run_dir,
            split_name="train",
            model=model,
            loader=data.train_eval_loader,
            class_names=data.class_names,
            device=device,
            topk=prediction_topk,
        )
        dump_vlm_adapter_predictions(
            run_dir=run_dir,
            split_name="val",
            model=model,
            loader=data.val_loader,
            class_names=data.class_names,
            device=device,
            topk=prediction_topk,
        )
        dump_vlm_adapter_predictions(
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
            "adapter_name": adapter_name,
            "initialization": initialization,
            "seed": seed,
            "best_epoch": best_state["best_epoch"],
            "zero_shot_test": zero_shot_test,
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
                "best_ckpt_file": "best_adapter.pt",
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
    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--local_model_path", type=str, default="./models/clip-vit-b32")
    parser.add_argument("--data_root", type=str, default="./datasets")

    parser.add_argument("--adapter_name", type=str, default="LP")
    parser.add_argument("--initialization", type=str, default="MEAN")
    parser.add_argument("--shots_per_class", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--method_name", type=str, default="vlm_adapter")
    parser.add_argument("--prediction_topk", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--hessian_dir", type=str, default=None)
    parser.add_argument("--pseudo_data_count", type=int, default=4)

    parser.add_argument("--taskres_alpha", type=float, default=0.5)
    parser.add_argument("--clipa_ratio", type=float, default=0.2)
    parser.add_argument("--clipa_hidden_dim", type=int, default=0)
    parser.add_argument("--tipa_alpha", type=float, default=1.0)
    parser.add_argument("--tipa_beta", type=float, default=1.0)
    parser.add_argument("--gaussian_prior_sigma", type=float, default=0.01)
    parser.add_argument("--gaussian_mc_samples", type=int, default=3)
    parser.add_argument("--gaussian_anneal_start_epoch", type=int, default=20)

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model_str=args.model,
        local_model_path=args.local_model_path,
        data_root=args.data_root,
        adapter_name=args.adapter_name,
        initialization=args.initialization,
        shots_per_class=args.shots_per_class,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        method_name=args.method_name,
        seed=args.seed,
        device=args.device,
        prediction_topk=args.prediction_topk,
        hessian_dir=args.hessian_dir,
        pseudo_data_count=args.pseudo_data_count,
        taskres_alpha=args.taskres_alpha,
        clipa_ratio=args.clipa_ratio,
        clipa_hidden_dim=args.clipa_hidden_dim,
        tipa_alpha=args.tipa_alpha,
        tipa_beta=args.tipa_beta,
        gaussian_prior_sigma=args.gaussian_prior_sigma,
        gaussian_mc_samples=args.gaussian_mc_samples,
        gaussian_anneal_start_epoch=args.gaussian_anneal_start_epoch,
    )