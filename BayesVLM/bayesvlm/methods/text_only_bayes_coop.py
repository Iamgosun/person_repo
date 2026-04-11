from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.distributions as dists
from torch.utils.data import DataLoader

from bayesvlm.coop_prompt import CoOpPromptLearner
from bayesvlm.hessians import KroneckerFactorizedCovariance
from bayesvlm.text_only_bayes_coop import TextOnlyBayesCoOpModel
from bayesvlm.training.io import save_jsonl
from bayesvlm.common import ProbabilisticLogits
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


def compute_text_covariance(
    A_txt: torch.Tensor,
    B_txt: torch.Tensor,
    n_txt: float,
    lambda_txt: float,
) -> KroneckerFactorizedCovariance:
    sqrt_n = math.sqrt(float(n_txt))
    sqrt_lambda = math.sqrt(float(lambda_txt))

    A = A_txt * sqrt_n + sqrt_lambda * torch.eye(
        A_txt.size(0),
        device=A_txt.device,
        dtype=A_txt.dtype,
    )
    B = B_txt * sqrt_n + sqrt_lambda * torch.eye(
        B_txt.size(0),
        device=B_txt.device,
        dtype=B_txt.dtype,
    )

    return KroneckerFactorizedCovariance(
        A_inv=torch.linalg.inv(A),
        B_inv=torch.linalg.inv(B),
    )


def build_text_only_bayes_coop_model(
    *,
    class_names: list[str],
    text_encoder: Any,
    image_encoder: Any,
    vlm: Any,
    text_covariance: KroneckerFactorizedCovariance,
    n_ctx: int,
    ctx_init: str,
    csc: bool,
    class_token_position: str,
    use_full_cov: bool,
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

    model = TextOnlyBayesCoOpModel(
        image_encoder=image_encoder,
        prompt_learner=prompt_learner,
        text_covariance=text_covariance,
        logit_scale=vlm.logit_scale,
        logit_bias=getattr(vlm, "logit_bias", None),
        use_full_cov=use_full_cov,
    ).to(device)

    return prompt_learner, model


def compute_text_only_bayes_coop_train_losses(
    *,
    model: TextOnlyBayesCoOpModel,
    prompt_learner: CoOpPromptLearner,
    batch: dict[str, Any],
    labels: torch.Tensor,
    ctx_anchor: torch.Tensor | None,
    ctx_reg_weight: float,
) -> dict[str, torch.Tensor]:
    map_logits = model.forward_map_logits(batch=batch)
    map_loss = torch.nn.functional.cross_entropy(
        map_logits,
        labels,
        reduction="mean",
    )

    bayes_logits = model.forward_bayes_logits(batch=batch)
    bayes_loss = bayes_logits.cross_entropy(
        labels,
        num_samples=0,
        reduction="mean",
    )

    if ctx_anchor is None or ctx_reg_weight <= 0:
        ctx_reg = map_loss.new_zeros(())
    else:
        ctx_reg = ((prompt_learner.ctx - ctx_anchor.to(prompt_learner.ctx.device)) ** 2).mean()

    return {
        "map_loss": map_loss,
        "bayes_loss": bayes_loss,
        "ctx_reg": ctx_reg,
    }


@torch.no_grad()
def _prepare_text_only_bayes_eval_cache(model: Any) -> dict[str, torch.Tensor | bool | None]:
    model.eval()

    mu, _, alpha, trace_sigma = model.compute_text_statistics()
    mu = mu.float()
    alpha = alpha.float()
    trace_sigma = trace_sigma.float()

    B_inv = model.text_covariance.B_inv.to(mu.device).float()
    mu_norm2 = (mu ** 2).sum(dim=-1).clamp_min(1e-6)
    denom_text = torch.sqrt(mu_norm2 + trace_sigma + 1e-6)

    cache = {
        "mu": mu,
        "mu_t": mu.t().contiguous(),
        "alpha": alpha,
        "trace_sigma": trace_sigma,
        "mu_norm2": mu_norm2,
        "denom_text": denom_text,
        "scale": model.logit_scale.exp().float(),
        "logit_bias": None if model.logit_bias is None else model.logit_bias.float(),
        "use_full_cov": bool(model.use_full_cov),
    }

    if model.use_full_cov:
        cache["B_inv"] = B_inv
    else:
        cache["diag_B"] = torch.diagonal(B_inv)

    return cache


@torch.no_grad()
def evaluate_text_only_bayes_coop(
    model: Any,
    loader: DataLoader,
    num_classes: int,
    device: str,
) -> dict[str, float]:
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    cache = _prepare_text_only_bayes_eval_cache(model)

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

        all_probs.append(probs)
        all_labels.append(labels)

        total_loss += prob_logits.cross_entropy(
            labels,
            num_samples=0,
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
def collect_text_only_bayes_coop_predictions(
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
    all_logits_mean = []
    all_logits_var = []

    sample_index = 0

    cache = _prepare_text_only_bayes_eval_cache(model)

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

        k = min(topk, probs.shape[1])
        topk_probs, topk_ids = torch.topk(probs, k=k, dim=1)

        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_logits_mean.append(prob_logits.mean.detach().cpu())
        all_logits_var.append(prob_logits.var.detach().cpu())

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
                "pred_logit_mean": float(prob_logits.mean[i, pred_id].item()),
                "pred_logit_var": float(prob_logits.var[i, pred_id].item()),
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
        "logits_mean": torch.cat(all_logits_mean, dim=0),
        "logits_var": torch.cat(all_logits_var, dim=0),
    }

    return rows, tensor_payload


def dump_text_only_bayes_coop_predictions(
    run_dir: Path,
    split_name: str,
    model: Any,
    loader: DataLoader,
    class_names: list[str],
    device: str,
    topk: int = 5,
) -> None:
    rows, tensor_payload = collect_text_only_bayes_coop_predictions(
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
