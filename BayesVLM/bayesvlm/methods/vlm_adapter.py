from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bayesvlm.training.io import save_jsonl
from bayesvlm.training.metrics import evaluate_prediction
from bayesvlm.vlm_adapter import VLMAdapter


def _get_batch_item(batch, key: str, idx: int, default=None):
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


def build_vlm_adapter_model(
    *,
    cfg: Any,
    class_names: list[str],
    image_encoder: Any,
    text_encoder: Any,
    vlm: Any,
    device: str,
):
    model = VLMAdapter(
        cfg=cfg,
        classnames=class_names,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        vlm=vlm,
    ).to(device)
    return model


@torch.no_grad()
def _collect_adapter_init_features(
    model: Any,
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
    model: Any,
    adapter_name: str,
    train_eval_loader: DataLoader,
    device: str,
) -> None:
    adapter_key = adapter_name.upper()
    if adapter_key == "TIPA" and hasattr(model.adapter, "init_tipadapter"):
        print("[TipA] 初始化 cache_keys / cache_values")
        train_features, train_labels = _collect_adapter_init_features(
            model=model,
            loader=train_eval_loader,
            device=device,
        )
        model.adapter.init_tipadapter(train_features, train_labels)


def compute_adapter_regularization_loss(model: Any) -> tuple[torch.Tensor, dict[str, float]]:
    if hasattr(model, "adapter_regularization_loss"):
        return model.adapter_regularization_loss()

    zero = torch.zeros((), device=model.logit_scale.device, dtype=torch.float32)
    return zero, {}


def compute_crossmodal_text_loss(
    model: Any,
    batch_size: int,
    device: str,
) -> torch.Tensor:
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
def evaluate_vlm_adapter(
    model: Any,
    loader: DataLoader,
    num_classes: int,
    device: str,
) -> dict[str, float]:
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
        "loss": float(total_loss / len(loader.dataset)),
        "acc": float(acc),
        "nlpd": float(nlpd),
        "ece": float(ece),
    }


@torch.no_grad()
def evaluate_zero_shot_vlm_adapter(
    model: Any,
    loader: DataLoader,
    num_classes: int,
    device: str,
) -> dict[str, float]:
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
        "loss": float(total_loss / len(loader.dataset)),
        "acc": float(acc),
        "nlpd": float(nlpd),
        "ece": float(ece),
    }


@torch.no_grad()
def collect_vlm_adapter_predictions(
    model: Any,
    loader: DataLoader,
    class_names: list[str],
    device: str,
    split_name: str,
    topk: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                "image_id": _get_batch_item(batch, "image_id", i, default=sample_index),
                "image_path": _get_batch_item(batch, "image_path", i, default=None),
                "text": _get_batch_item(batch, "text", i, default=None),
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


def dump_vlm_adapter_predictions(
    run_dir: Path,
    split_name: str,
    model: Any,
    loader: DataLoader,
    class_names: list[str],
    device: str,
    topk: int = 5,
) -> None:
    rows, tensor_payload = collect_vlm_adapter_predictions(
        model=model,
        loader=loader,
        class_names=class_names,
        device=device,
        split_name=split_name,
        topk=topk,
    )

    save_jsonl(run_dir / f"{split_name}_predictions.jsonl", rows)
    torch.save(tensor_payload, run_dir / f"{split_name}_predictions.pt")