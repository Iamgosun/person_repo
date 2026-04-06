from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassCalibrationError

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.utils import get_image_size, get_model_type_and_size, get_transform, load_model
from bayesvlm.vlm_adapter import VLMAdapter


SUPPORTED_DATASETS = [
    "flowers102",
    "food101",
    "cifar10",
    "cifar100",
    "imagenet-r",
    "ucf101",
    "sun397",
]


class Config:
    pass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_dataset(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def get_class_names(ds):
    base_ds = unwrap_dataset(ds)

    class_names = getattr(base_ds, "classes", None)
    if class_names is None:
        class_names = getattr(base_ds, "_label_names", None)
    if class_names is None:
        class_names = getattr(base_ds, "label_names", None)
    if class_names is None:
        class_names = getattr(base_ds, "classnames", None)

    if class_names is None:
        raise ValueError("当前数据集无法自动提取类别名，请手动补充。")

    return list(class_names)


def print_class_counts(ds, split_name="train"):
    class_names = get_class_names(ds)
    counter = Counter()

    for i in range(len(ds)):
        sample = ds[i]
        counter[int(sample["class_id"])] += 1

    print(f"===== {split_name} =====")
    print(f"num_classes: {len(class_names)}")
    for class_id in sorted(counter.keys()):
        print(f"{class_id:3d} | {class_names[class_id]:25s} | {counter[class_id]}")

    return class_names, counter


def build_fewshot_subset(ds, shots_per_class: int, seed: int):
    """
    和 train_text_only_bayes_coop 保持一致：
    从训练集按每类 K-shot 抽样；如果 shots_per_class <= 0，则直接返回原训练集。
    """
    if shots_per_class <= 0:
        return ds

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    picked = defaultdict(list)
    for idx in indices:
        label = int(ds[idx]["class_id"])
        if len(picked[label]) < shots_per_class:
            picked[label].append(idx)

    final_indices = []
    for label in sorted(picked.keys()):
        final_indices.extend(picked[label])

    final_indices = sorted(final_indices)
    print(f"[few-shot] 训练集从 {len(ds)} 条样本，抽成 {len(final_indices)} 条样本")
    return Subset(ds, final_indices)


def build_loader(ds, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics_history(metrics_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in metrics_history:
        row = {
            "epoch": item["epoch"],
            "train_loss_step_mean": item["train_loss_step_mean"],
        }
        for split in ["train", "val", "test"]:
            for key, value in item[split].items():
                row[f"{split}_{key}"] = value

        for extra_key in ["loss_kl", "loss_crossmodal_text"]:
            if extra_key in item:
                row[extra_key] = item[extra_key]

        rows.append(row)
    return rows


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr

    with open(log_path, "w", encoding="utf-8") as f:
        tee = Tee(stdout_backup, f)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = stdout_backup
            sys.stderr = stderr_backup


def get_batch_item(batch, key: str, idx: int, default=None):
    if key not in batch:
        return default

    value = batch[key]

    if torch.is_tensor(value):
        item = value[idx]
        if item.ndim == 0:
            return item.item()
        return item.detach().cpu().tolist()

    if isinstance(value, (list, tuple)):
        item = value[idx]
        if torch.is_tensor(item):
            if item.ndim == 0:
                return item.item()
            return item.detach().cpu().tolist()
        return item

    return value


@torch.no_grad()
def evaluate_prediction(prediction: torch.Tensor, label: torch.Tensor, num_classes: int):
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm="l1")
    pred_cls = prediction.argmax(1)
    acc = (pred_cls == label).float().mean().item()
    nlpd = -dists.Categorical(prediction).log_prob(label).mean().item()
    ece = ece_metric(prediction, label).item()
    return acc, nlpd, ece


@torch.no_grad()
def evaluate_zero_shot(
    model: VLMAdapter,
    loader: DataLoader,
    num_classes: int,
    device: str,
) -> Dict[str, float]:
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        labels = batch["class_id"].to(device)
        logits = model.zero_shot_logits(batch=batch)
        probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc, nlpd, ece = evaluate_prediction(all_probs, all_labels, num_classes=num_classes)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": acc,
        "nlpd": nlpd,
        "ece": ece,
    }


@torch.no_grad()
def evaluate_model(
    model: VLMAdapter,
    loader: DataLoader,
    num_classes: int,
    device: str,
) -> Dict[str, float]:
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        labels = batch["class_id"].to(device)
        logits = model(batch=batch)
        probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc, nlpd, ece = evaluate_prediction(all_probs, all_labels, num_classes=num_classes)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": acc,
        "nlpd": nlpd,
        "ece": ece,
    }


@torch.no_grad()
def collect_adapter_init_features(
    model: VLMAdapter,
    loader: DataLoader,
    device: str,
):
    model.eval()
    all_features = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["class_id"].to(device)
        features = model._encode_image(images)
        all_features.append(features.detach().cpu())
        all_labels.append(labels.detach().cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def maybe_init_special_adapter(
    model: VLMAdapter,
    adapter_name: str,
    train_eval_loader: DataLoader,
    device: str,
):
    adapter_key = adapter_name.upper()
    if adapter_key == "TIPA" and hasattr(model.adapter, "init_tipadapter"):
        print("[TipA] 初始化 cache_keys / cache_values")
        train_features, train_labels = collect_adapter_init_features(
            model=model,
            loader=train_eval_loader,
            device=device,
        )
        model.adapter.init_tipadapter(train_features, train_labels)


def compute_regularization_loss(model: VLMAdapter, adapter_name: str):
    adapter_key = adapter_name.upper()
    aux: Dict[str, float] = {}
    zero = torch.zeros((), device=model.logit_scale.device, dtype=torch.float32)

    if adapter_key == "GAUSSIAN_PER_CLASS" and hasattr(model.adapter, "kl_divergence"):
        kl = model.adapter.kl_divergence()
        aux["loss_kl"] = float(kl.detach().item())
        return kl, aux

    return zero, aux


def compute_crossmodal_text_loss(
    model: VLMAdapter,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    保留 CrossModal 的“文本 prototype 辅助监督”思想，
    但不再把主训练流程改成 feature-dataset。
    """
    text_proto = model.base_text_features.to(device=device, dtype=torch.float32)
    num_classes = text_proto.shape[0]

    sampled_labels = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size,),
        device=device,
    )
    sampled_features = text_proto[sampled_labels]
    logits = model.forward_features(sampled_features)
    return F.cross_entropy(logits, sampled_labels)


@torch.no_grad()
def collect_predictions(
    model: VLMAdapter,
    loader: DataLoader,
    class_names: List[str],
    device: str,
    split_name: str,
    topk: int = 5,
):
    model.eval()

    rows = []
    all_labels = []
    all_preds = []
    all_probs = []
    all_logits = []

    sample_index = 0

    for batch in loader:
        labels = batch["class_id"].to(device)
        logits = model(batch=batch)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=1)

        k = min(topk, probs.shape[1])
        topk_probs, topk_ids = torch.topk(probs, k=k, dim=1)

        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_logits.append(logits.detach().cpu())

        for i in range(labels.shape[0]):
            label_id = int(labels[i].item())
            pred_id = int(preds[i].item())

            topk_list = []
            for rank in range(k):
                class_id = int(topk_ids[i, rank].item())
                topk_list.append(
                    {
                        "rank": rank + 1,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "prob": float(topk_probs[i, rank].item()),
                    }
                )

            row = {
                "split": split_name,
                "sample_index": sample_index,
                "image_id": get_batch_item(batch, "image_id", i, default=sample_index),
                "image_path": get_batch_item(batch, "image_path", i, default=None),
                "text": get_batch_item(batch, "text", i, default=None),
                "label_id": label_id,
                "label_name": class_names[label_id],
                "pred_id": pred_id,
                "pred_name": class_names[pred_id],
                "confidence": float(probs[i, pred_id].item()),
                "correct": bool(label_id == pred_id),
                "topk": topk_list,
            }
            rows.append(row)
            sample_index += 1

    tensor_payload = {
        "split": split_name,
        "class_names": class_names,
        "labels": torch.cat(all_labels, dim=0),
        "preds": torch.cat(all_preds, dim=0),
        "probs": torch.cat(all_probs, dim=0),
        "logits": torch.cat(all_logits, dim=0),
    }

    return rows, tensor_payload


def dump_predictions(
    run_dir: Path,
    split_name: str,
    model: VLMAdapter,
    loader: DataLoader,
    class_names: List[str],
    device: str,
    topk: int = 5,
):
    rows, tensor_payload = collect_predictions(
        model=model,
        loader=loader,
        class_names=class_names,
        device=device,
        split_name=split_name,
        topk=topk,
    )

    save_jsonl(run_dir / f"{split_name}_predictions.jsonl", rows)
    torch.save(tensor_payload, run_dir / f"{split_name}_predictions.pt")


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
):
    set_seed(seed)

    run_dir = (
        Path(save_dir)
        / method_name
        / dataset
        / adapter_name.upper()
        / initialization
        / f"shot_{shots_per_class}"
        / f"seed_{seed}"
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
        print(f"[run] run_dir={run_dir}")

        if hessian_dir:
            print(f"[note] 当前 vlm_adapter 仍是确定性对比实验，hessian_dir={hessian_dir} 将被忽略。")
        print(f"[note] pseudo_data_count={pseudo_data_count} 仅为兼容旧接口保留，不参与当前 adapter 主训练。")

        if model_str not in MODEL_NAME_MAP:
            raise ValueError(f"无效模型名：{model_str}，可选值为 {list(MODEL_NAME_MAP.keys())}")

        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"无效数据集：{dataset}，可选值为 {SUPPORTED_DATASETS}")

        model_type, _ = get_model_type_and_size(model_str)
        transform_image_size = get_image_size(model_str)
        transform = get_transform(model_type, transform_image_size)

        factory = DataModuleFactory(
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=transform,
            test_transform=transform,
            shuffle_train=True,
            base_path=data_root,
        )
        dm = factory.create(dataset)
        dm.setup()

        train_ds = build_fewshot_subset(dm.train_ds, shots_per_class=shots_per_class, seed=seed)
        val_ds = dm.val_ds if hasattr(dm, "val_ds") and dm.val_ds is not None else dm.test_ds
        test_ds = dm.test_ds

        class_names, _ = print_class_counts(train_ds, split_name="train")
        print_class_counts(test_ds, split_name="test")
        save_json(run_dir / "class_names.json", {"class_names": class_names})

        # 关键修改：直接复用 text_only_bayes_coop 的 raw-image loader protocol
        train_loader = build_loader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        train_eval_loader = build_loader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        val_loader = build_loader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        test_loader = build_loader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        image_encoder, text_encoder, vlm = load_model(
            model_str=model_str,
            device=device,
            local_model_path=local_model_path,
        )

        cfg = Config()
        cfg.model = model_str
        cfg.model_name_or_path = local_model_path
        cfg.datasetname = dataset
        cfg.adapter_name = adapter_name
        cfg.initialization = initialization
        cfg.device = device

        model = VLMAdapter(
            cfg=cfg,
            classnames=class_names,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            vlm=vlm,
        ).to(device)

        maybe_init_special_adapter(
            model=model,
            adapter_name=adapter_name,
            train_eval_loader=train_eval_loader,
            device=device,
        )

        zero_shot_test = evaluate_zero_shot(
            model=model,
            loader=test_loader,
            num_classes=len(class_names),
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

            epoch_loss_sum = 0.0
            epoch_count = 0
            epoch_crossmodal_text_sum = 0.0

            for batch in train_loader:
                labels = batch["class_id"].to(device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(batch=batch)
                loss = F.cross_entropy(logits, labels)

                reg_loss, reg_info = compute_regularization_loss(model, adapter_name=adapter_name)
                total_loss = loss + reg_loss

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
                epoch_count += labels.size(0)

            train_metrics = evaluate_model(model, train_eval_loader, len(class_names), device=device)
            val_metrics = evaluate_model(model, val_loader, len(class_names), device=device)
            test_metrics = evaluate_model(model, test_loader, len(class_names), device=device)

            row = {
                "epoch": epoch,
                "train_loss_step_mean": epoch_loss_sum / max(epoch_count, 1),
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            }

            if adapter_name.upper() == "GAUSSIAN_PER_CLASS" and hasattr(model.adapter, "kl_divergence"):
                row["loss_kl"] = float(model.adapter.kl_divergence().detach().item())

            if adapter_name.upper() == "CROSSMODAL":
                row["loss_crossmodal_text"] = epoch_crossmodal_text_sum / max(epoch_count, 1)

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

        final_train_metrics = evaluate_model(model, train_eval_loader, len(class_names), device=device)
        final_val_metrics = evaluate_model(model, val_loader, len(class_names), device=device)
        final_test_metrics = evaluate_model(model, test_loader, len(class_names), device=device)

        dump_predictions(
            run_dir=run_dir,
            split_name="train",
            model=model,
            loader=train_eval_loader,
            class_names=class_names,
            device=device,
            topk=prediction_topk,
        )
        dump_predictions(
            run_dir=run_dir,
            split_name="val",
            model=model,
            loader=val_loader,
            class_names=class_names,
            device=device,
            topk=prediction_topk,
        )
        dump_predictions(
            run_dir=run_dir,
            split_name="test",
            model=model,
            loader=test_loader,
            class_names=class_names,
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

    # 仅为兼容旧接口保留，不参与当前 adapter 主训练。
    parser.add_argument("--hessian_dir", type=str, default=None)
    parser.add_argument("--pseudo_data_count", type=int, default=4)

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
    )