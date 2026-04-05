from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def tee_output(log_path: Path):
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    import sys

    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as f:
        tee = _Tee(original_stdout, f)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


class DictSubset(Dataset):
    def __init__(self, dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class FeatureDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        image_ids: torch.Tensor | None = None,
        image_paths: List[str] | None = None,
        source: str = "image",
    ):
        self.features = features.float().cpu()
        self.labels = labels.long().cpu()
        self.image_ids = image_ids.long().cpu() if image_ids is not None else None
        self.image_paths = list(image_paths) if image_paths is not None else None
        self.source = str(source)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        item = {
            "features": self.features[idx],
            "class_id": self.labels[idx],
            "source": self.source,
        }
        if self.image_ids is not None:
            item["image_id"] = self.image_ids[idx]
        if self.image_paths is not None:
            item["image_path"] = self.image_paths[idx]
        return item


class ConcatFeatureDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        self.boundaries = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.boundaries.append(total)

    def __len__(self):
        return self.boundaries[-1] if self.boundaries else 0

    def __getitem__(self, idx):
        for ds_idx, boundary in enumerate(self.boundaries):
            if idx < boundary:
                prev = 0 if ds_idx == 0 else self.boundaries[ds_idx - 1]
                return self.datasets[ds_idx][idx - prev]
        raise IndexError(idx)


class Config:
    pass


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def unwrap_dataset(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def get_class_names(ds=None, dm=None) -> List[str]:
    if dm is not None:
        label_names = getattr(dm, "label_names", None)
        if label_names is not None:
            return list(label_names)

        class_prompts = getattr(dm, "class_prompts", None)
        if class_prompts is not None:
            class_prompts = list(class_prompts)
            if class_prompts:
                prefix = "An image of a "
                names = []
                for prompt in class_prompts:
                    if isinstance(prompt, str) and prompt.startswith(prefix):
                        names.append(prompt[len(prefix):])
                    else:
                        names.append(str(prompt))
                return names

        if ds is None:
            ds = getattr(dm, "train_ds", None)

    if ds is None:
        raise ValueError("get_class_names() 需要 ds 或 dm。")

    base_ds = unwrap_dataset(ds)
    for attr in ["classes", "_label_names", "label_names", "classnames"]:
        class_names = getattr(base_ds, attr, None)
        if class_names is not None:
            return list(class_names)

    raise ValueError(f"当前数据集无法自动提取类别名: dataset_type={type(base_ds)}")


def print_class_counts(ds, split_name: str = "train", class_names: List[str] | None = None):
    if class_names is None:
        class_names = get_class_names(ds=ds)
    counter = Counter()
    for i in range(len(ds)):
        counter[int(ds[i]["class_id"])] += 1

    print(f"===== {split_name} =====")
    print(f"num_classes: {len(class_names)}")
    for class_id in sorted(counter.keys()):
        print(f"{class_id:3d} | {class_names[class_id]:25s} | {counter[class_id]}")
    return class_names, counter


def build_fewshot_subset(ds, shots_per_class: int, seed: int):
    if shots_per_class <= 0:
        raise ValueError("few-shot adapter 对比实验里，shots_per_class 必须 > 0；zero-shot 请单独看冻结 CLIP 指标。")

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
    return DictSubset(ds, final_indices)


@torch.no_grad()
def evaluate_prediction(prediction: torch.Tensor, label: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm="l1")
    pred_cls = prediction.argmax(1)
    acc = (pred_cls == label).float().mean().item()
    nlpd = -dists.Categorical(prediction).log_prob(label).mean().item()
    ece = ece_metric(prediction, label).item()
    return acc, nlpd, ece


@torch.no_grad()
def extract_image_features(model: VLMAdapter, dataset, batch_size: int, num_workers: int, device: str, shuffle: bool = False):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    model.eval()
    features = []
    labels = []
    image_ids = []
    image_paths = []
    for batch in loader:
        images = batch["image"].to(device)
        feats = model._encode_image(images)
        features.append(feats.detach().cpu())
        labels.append(batch["class_id"].detach().cpu())
        if "image_id" in batch:
            image_ids.append(batch["image_id"].detach().cpu())
        if "image_path" in batch:
            image_paths.extend(list(batch["image_path"]))
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    image_ids_tensor = torch.cat(image_ids, dim=0) if image_ids else None
    image_paths_list = image_paths if image_paths else None
    return features, labels, image_ids_tensor, image_paths_list


@torch.no_grad()
def build_zero_shot_probs(model: VLMAdapter, features: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    rows = []
    model.eval()
    for start in range(0, features.shape[0], batch_size):
        feats = features[start : start + batch_size].to(device)
        image_features = feats.to(device=device, dtype=torch.float32)
        text_features = model.base_text_features.to(device=image_features.device, dtype=image_features.dtype)
        logits = model.vlm(image_features, text_features)
        if getattr(model.vlm, "logit_bias", None) is not None:
            logits = logits + model.vlm.logit_bias.to(device=logits.device, dtype=logits.dtype)
        rows.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def evaluate_zero_shot(model: VLMAdapter, features: torch.Tensor, labels: torch.Tensor, batch_size: int, device: str):
    probs = build_zero_shot_probs(model, features, batch_size=batch_size, device=device)
    acc, nlpd, ece = evaluate_prediction(probs, labels.cpu(), num_classes=probs.shape[1])
    return {"acc": acc, "nlpd": nlpd, "ece": ece}


@torch.no_grad()
def build_adapter_feature_dataset(
    adapter_name: str,
    model: VLMAdapter,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_image_ids: torch.Tensor | None,
    train_image_paths: List[str] | None,
    seed: int,
) -> Dataset:
    adapter_key = adapter_name.upper()
    base_ds = FeatureDataset(
        features=train_features,
        labels=train_labels,
        image_ids=train_image_ids,
        image_paths=train_image_paths,
        source="image",
    )

    if adapter_key != "CROSSMODAL":
        return base_ds

    rng = np.random.default_rng(seed)
    text_proto = model.base_text_features.detach().cpu()
    n_classes = text_proto.shape[0]
    per_class_views = 1
    all_text = []
    all_labels = []
    for c in range(n_classes):
        for _ in range(per_class_views):
            all_text.append(text_proto[c])
            all_labels.append(c)
    text_features = torch.stack(all_text, dim=0)
    text_labels = torch.tensor(all_labels, dtype=torch.long)

    if text_features.shape[0] < train_features.shape[0]:
        idx = rng.choice(text_features.shape[0], size=train_features.shape[0], replace=True)
        text_features = text_features[idx]
        text_labels = text_labels[idx]

    text_ds = FeatureDataset(
        features=text_features,
        labels=text_labels,
        image_ids=None,
        image_paths=None,
        source="text",
    )
    print(f"[CrossModal] 额外混入 {len(text_ds)} 条文本 prototype 样本")
    return ConcatFeatureDataset([base_ds, text_ds])


def make_feature_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def evaluate_model_on_features(model: VLMAdapter, loader: DataLoader, num_classes: int, device: str) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0

    for batch in loader:
        labels = batch["class_id"].to(device)
        feats = batch["features"].to(device)
        logits = model.forward_features(feats)
        probs = torch.softmax(logits, dim=-1)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    acc, nlpd, ece = evaluate_prediction(all_probs, all_labels, num_classes=num_classes)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "acc": acc,
        "nlpd": nlpd,
        "ece": ece,
    }


def compute_regularization_loss(model: VLMAdapter, adapter_name: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    adapter_key = adapter_name.upper()
    aux: Dict[str, float] = {}
    zero = torch.zeros((), device=model.logit_scale.device, dtype=torch.float32)

    if adapter_key == "GAUSSIAN_PER_CLASS" and hasattr(model.adapter, "kl_divergence"):
        kl = model.adapter.kl_divergence()
        aux["loss_kl"] = float(kl.detach().item())
        return kl, aux

    return zero, aux


@torch.no_grad()
def collect_predictions(
    model: VLMAdapter,
    loader: DataLoader,
    class_names: Sequence[str],
    device: str,
    split_name: str,
    topk: int,
):
    model.eval()
    rows: List[Dict[str, Any]] = []
    all_probs = []
    all_labels = []
    all_topk_indices = []
    all_topk_scores = []

    for batch in loader:
        feats = batch["features"].to(device)
        labels = batch["class_id"].to(device)
        logits = model.forward_features(feats)
        probs = torch.softmax(logits, dim=-1)
        scores, indices = torch.topk(probs, k=min(topk, probs.shape[1]), dim=-1)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_topk_indices.append(indices.cpu())
        all_topk_scores.append(scores.cpu())

        image_ids = batch.get("image_id", None)
        image_paths = batch.get("image_path", None)
        sources = batch.get("source", None)

        for i in range(probs.shape[0]):
            label_id = int(labels[i].item())
            pred_id = int(indices[i, 0].item())
            row = {
                "split": split_name,
                "label_id": label_id,
                "label_name": class_names[label_id],
                "pred_id": pred_id,
                "pred_name": class_names[pred_id],
                "pred_prob": float(scores[i, 0].item()),
            }
            if image_ids is not None:
                row["image_id"] = int(image_ids[i].item())
            if image_paths is not None:
                row["image_path"] = image_paths[i]
            if sources is not None:
                row["source"] = sources[i]
            for k in range(indices.shape[1]):
                row[f"top{k+1}_id"] = int(indices[i, k].item())
                row[f"top{k+1}_name"] = class_names[int(indices[i, k].item())]
                row[f"top{k+1}_prob"] = float(scores[i, k].item())
            rows.append(row)

    tensor_payload = {
        "probabilities": torch.cat(all_probs, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "topk_indices": torch.cat(all_topk_indices, dim=0),
        "topk_scores": torch.cat(all_topk_scores, dim=0),
    }
    return rows, tensor_payload


def dump_predictions(
    run_dir: Path,
    split_name: str,
    model: VLMAdapter,
    loader: DataLoader,
    class_names: Sequence[str],
    device: str,
    topk: int,
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


@torch.no_grad()
def maybe_init_special_adapter(model: VLMAdapter, adapter_name: str, train_features: torch.Tensor, train_labels: torch.Tensor):
    adapter_key = adapter_name.upper()
    if adapter_key == "TIPA" and hasattr(model.adapter, "init_tipadapter"):
        print("[TipA] 初始化 cache_keys / cache_values")
        model.adapter.init_tipadapter(train_features, train_labels)



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

    run_dir = Path(save_dir) / method_name / dataset / adapter_name.upper() / initialization / f"shot_{shots_per_class}" / f"seed_{seed}"
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
            print(f"[note] 当前 train.py 为确定性 adapter 训练器，hessian_dir={hessian_dir} 将被忽略。")
        print(f"[note] pseudo_data_count={pseudo_data_count} 仅为兼容旧接口保留，不参与本训练。")

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

        train_features, train_labels, train_image_ids, train_image_paths = extract_image_features(
            model=model,
            dataset=train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            shuffle=False,
        )
        val_features, val_labels, val_image_ids, val_image_paths = extract_image_features(
            model=model,
            dataset=val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            shuffle=False,
        )
        test_features, test_labels, test_image_ids, test_image_paths = extract_image_features(
            model=model,
            dataset=test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            shuffle=False,
        )

        maybe_init_special_adapter(model, adapter_name=adapter_name, train_features=train_features, train_labels=train_labels)

        train_feature_ds = build_adapter_feature_dataset(
            adapter_name=adapter_name,
            model=model,
            train_features=train_features,
            train_labels=train_labels,
            train_image_ids=train_image_ids,
            train_image_paths=train_image_paths,
            seed=seed,
        )
        train_eval_feature_ds = FeatureDataset(train_features, train_labels, train_image_ids, train_image_paths, source="image")
        val_feature_ds = FeatureDataset(val_features, val_labels, val_image_ids, val_image_paths, source="image")
        test_feature_ds = FeatureDataset(test_features, test_labels, test_image_ids, test_image_paths, source="image")

        train_loader = make_feature_loader(train_feature_ds, batch_size=batch_size, shuffle=True)
        train_eval_loader = make_feature_loader(train_eval_feature_ds, batch_size=batch_size, shuffle=False)
        val_loader = make_feature_loader(val_feature_ds, batch_size=batch_size, shuffle=False)
        test_loader = make_feature_loader(test_feature_ds, batch_size=batch_size, shuffle=False)

        zero_shot_test = evaluate_zero_shot(model, test_features, test_labels, batch_size=batch_size, device=device)
        print(f"[zero-shot] test_acc={zero_shot_test['acc']:.4f} test_nlpd={zero_shot_test['nlpd']:.4f} test_ece={zero_shot_test['ece']:.4f}")

        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)

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
            "zero_shot_test": zero_shot_test,
        }
        save_json(run_dir / "config.json", config)

        best_val_loss = float("inf")
        best_state = None
        history: List[Dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss_sum = 0.0
            epoch_count = 0

            for batch in train_loader:
                feats = batch["features"].to(device)
                labels = batch["class_id"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.forward_features(feats)
                loss = F.cross_entropy(logits, labels)
                reg_loss, reg_info = compute_regularization_loss(model, adapter_name=adapter_name)
                total_loss = loss + reg_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss_sum += total_loss.item() * labels.size(0)
                epoch_count += labels.size(0)

            train_metrics = evaluate_model_on_features(model, train_eval_loader, len(class_names), device=device)
            val_metrics = evaluate_model_on_features(model, val_loader, len(class_names), device=device)
            test_metrics = evaluate_model_on_features(model, test_loader, len(class_names), device=device)

            row = {
                "epoch": epoch,
                "train_loss_step_mean": epoch_loss_sum / max(epoch_count, 1),
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_nlpd": train_metrics["nlpd"],
                "train_ece": train_metrics["ece"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_nlpd": val_metrics["nlpd"],
                "val_ece": val_metrics["ece"],
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["acc"],
                "test_nlpd": test_metrics["nlpd"],
                "test_ece": test_metrics["ece"],
            }
            if adapter_name.upper() == "GAUSSIAN_PER_CLASS" and hasattr(model.adapter, "kl_divergence"):
                row["loss_kl"] = float(model.adapter.kl_divergence().detach().item())
            history.append(row)

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

            save_json(run_dir / "metrics_history.json", history)
            save_csv(run_dir / "metrics_history.csv", history)

        if best_state is None:
            raise RuntimeError("训练未产生 best checkpoint")

        model.adapter.load_state_dict(best_state["adapter"])
        final_train_metrics = evaluate_model_on_features(model, train_eval_loader, len(class_names), device=device)
        final_val_metrics = evaluate_model_on_features(model, val_loader, len(class_names), device=device)
        final_test_metrics = evaluate_model_on_features(model, test_loader, len(class_names), device=device)

        dump_predictions(run_dir, "train", model, train_eval_loader, class_names, device, topk=prediction_topk)
        dump_predictions(run_dir, "val", model, val_loader, class_names, device, topk=prediction_topk)
        dump_predictions(run_dir, "test", model, test_loader, class_names, device, topk=prediction_topk)

        summary = {
            "method_name": method_name,
            "dataset": dataset,
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

    # 为了兼容旧 train.py / shell 脚本保留，但在确定性 adapter 版本里不使用。
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
