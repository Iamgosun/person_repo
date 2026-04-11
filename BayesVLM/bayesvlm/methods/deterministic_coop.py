from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.distributions as dists
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

from bayesvlm.coop_prompt import CoOpPromptLearner
from bayesvlm.deterministic_coop import DeterministicCoOpModel
from bayesvlm.training.io import save_jsonl
from bayesvlm.training.metrics import calculate_official_bayesadapter_ece


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


def build_deterministic_coop_model(
    *,
    class_names: list[str],
    text_encoder: Any,
    image_encoder: Any,
    vlm: Any,
    n_ctx: int,
    ctx_init: str,
    csc: bool,
    class_token_position: str,
    device: str,
):
    if hasattr(image_encoder, "freeze_all_layers"):
        image_encoder.freeze_all_layers()
    else:
        for p in image_encoder.parameters():
            p.requires_grad = False

    if hasattr(text_encoder, "freeze_all_layers"):
        text_encoder.freeze_all_layers()
    else:
        for p in text_encoder.parameters():
            p.requires_grad = False

    vlm.logit_scale.requires_grad = False
    if getattr(vlm, "logit_bias", None) is not None:
        vlm.logit_bias.requires_grad = False

    image_encoder.eval()
    text_encoder.eval()
    vlm.eval()

    prompt_learner = CoOpPromptLearner(
        class_names=class_names,
        text_encoder=text_encoder,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        csc=csc,
        class_token_position=class_token_position,
    ).to(device)

    model = DeterministicCoOpModel(
        image_encoder=image_encoder,
        prompt_learner=prompt_learner,
        vlm=vlm,
    ).to(device)

    return prompt_learner, model


@torch.no_grad()
def _prepare_deterministic_eval_cache(model: Any) -> dict[str, torch.Tensor]:
    model.eval()
    text_outputs = model.prompt_learner()
    text_features = text_outputs.embeds.float()
    return {
        "text_features": text_features,
    }


@torch.no_grad()
def evaluate_deterministic_coop(
    model: Any,
    loader: DataLoader,
    num_classes: int,
    device: str,
) -> dict[str, float]:
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    cache = _prepare_deterministic_eval_cache(model)
    text_features = cache["text_features"]

    for batch in loader:
        labels = batch["class_id"].to(device)
        g = model.encode_image_batch(batch=batch)
        logits = model.vlm(g, text_features)

        probs = F.softmax(logits, dim=-1)

        all_probs.append(probs)
        all_labels.append(labels)

        total_loss += F.cross_entropy(
            logits,
            labels,
            reduction="sum",
        ).item()

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds = all_probs.argmax(dim=1)
    acc = (preds == all_labels).float().mean().item()
    nlpd = -dists.Categorical(all_probs).log_prob(all_labels).mean().item()
    ece = calculate_official_bayesadapter_ece(all_probs, all_labels, n_bins=10)
    loss = total_loss / len(loader.dataset)

    return {
        "loss": float(loss),
        "acc": float(acc),
        "nlpd": float(nlpd),
        "ece": float(ece),
    }


@torch.no_grad()
def collect_deterministic_coop_predictions(
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

    cache = _prepare_deterministic_eval_cache(model)
    text_features = cache["text_features"]

    for batch in loader:
        labels = batch["class_id"].to(device)
        g = model.encode_image_batch(batch=batch)
        logits = model.vlm(g, text_features)
        probs = F.softmax(logits, dim=-1)
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
                "sample_key": _get_batch_item(batch, "sample_key", i, default=None),
                "image_path": _get_batch_item(batch, "image_path", i, default=None),
                "text": _get_batch_item(batch, "text", i, default=None),
                "label_id": label_id,
                "label_name": class_names[label_id],
                "pred_id": pred_id,
                "pred_name": class_names[pred_id],
                "confidence": float(probs[i, pred_id].item()),
                "correct": bool(label_id == pred_id),
                "pred_logit": float(logits[i, pred_id].item()),
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


def dump_deterministic_coop_predictions(
    run_dir: Path,
    split_name: str,
    model: Any,
    loader: DataLoader,
    class_names: list[str],
    device: str,
    topk: int = 5,
) -> None:
    rows, tensor_payload = collect_deterministic_coop_predictions(
        model=model,
        loader=loader,
        class_names=class_names,
        device=device,
        split_name=split_name,
        topk=topk,
    )

    split_dir = run_dir / "eval" / "id" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(split_dir / "predictions.jsonl", rows)
    torch.save(tensor_payload, split_dir / "predictions.pt")
