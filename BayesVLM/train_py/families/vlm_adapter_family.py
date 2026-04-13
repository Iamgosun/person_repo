from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from bayesvlm.hessians import load_hessians
from bayesvlm.methods.text_only_bayes_coop import build_text_only_bayes_coop_model, compute_text_covariance
from bayesvlm.methods.vlm_adapter import (
    build_vlm_adapter_model,
    collect_vlm_adapter_predictions,
    compute_adapter_regularization_loss,
    compute_classification_loss_from_logits,
    compute_crossmodal_text_loss,
    evaluate_vlm_adapter,
    evaluate_zero_shot_vlm_adapter,
    reduce_logits_for_inference,
)
from bayesvlm.text_priors import _build_templates
from train_py.families.base_family import BaseFamily
from train_py.runtime.common_context import resolve_existing_path
from train_py.train_runtime import build_optimizer_from_args


def _slugify_path_part(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "empty"
    chars = []
    for ch in text:
        chars.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(chars)


def _extract_prompt_state_dict(ckpt_obj: dict[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "prompt_learner" in ckpt_obj and isinstance(ckpt_obj["prompt_learner"], dict):
        return ckpt_obj["prompt_learner"]
    if isinstance(ckpt_obj, dict) and any(torch.is_tensor(v) for v in ckpt_obj.values()):
        return ckpt_obj
    raise ValueError("could not parse prompt_learner state_dict from checkpoint")


def _read_saved_class_names(run_dir: Path) -> list[str] | None:
    class_names_path = run_dir / "config" / "class_names.json"
    if not class_names_path.exists():
        return None
    with open(class_names_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "class_names" in data:
        return list(data["class_names"])
    if isinstance(data, list):
        return list(data)
    return None


def _resolve_text_only_ckpt_path(run_dir: Path, config: dict[str, Any], ckpt_spec: str) -> Path:
    spec = str(ckpt_spec or "auto").strip()
    if spec in {"", "auto"}:
        selection = str(config.get("model_selection", "last")).strip().lower()
        spec = "best" if selection == "best" else "last"
    checkpoints_dir = run_dir / "checkpoints"
    if spec == "best":
        ckpt_path = checkpoints_dir / "best_prompt_learner.pt"
    elif spec == "last":
        ckpt_path = checkpoints_dir / "last_prompt_learner.pt"
    else:
        candidate = Path(spec)
        ckpt_path = candidate if candidate.is_absolute() else (run_dir / candidate)
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"text_only checkpoint not found: {ckpt_path}")
    return ckpt_path


def _has_text_only_bridge(args) -> bool:
    run_dir_raw = str(getattr(args, "bayesadapter_text_only_run_dir", "")).strip()
    run_dir_template = str(getattr(args, "bayesadapter_text_only_run_dir_template", "")).strip()
    return bool(run_dir_raw or run_dir_template)


def _module_device(module, fallback: str = "cpu") -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device(fallback)


@torch.no_grad()
def _build_base_text_features_for_bridge(ctx, args) -> torch.Tensor:
    templates = _build_templates(str(args.dataset))
    device = _module_device(ctx.text_encoder, fallback=args.device)
    class_features = []
    ctx.text_encoder.eval()
    for class_name in ctx.class_names:
        prompts = [template.format(class_name.replace("_", " ")) for template in templates]
        text_embeds = ctx.text_encoder(prompts)
        if hasattr(text_embeds, "embeds"):
            text_embeds = text_embeds.embeds
        if isinstance(text_embeds, tuple):
            text_embeds = text_embeds[0]
        text_embeds = text_embeds.to(device=device, dtype=torch.float32)
        class_features.append(text_embeds.mean(dim=0))
    return torch.stack(class_features, dim=0)


@torch.no_grad()
def _resolve_bayesadapter_canonical_prior(ctx, args, source_payload: dict[str, Any] | None) -> dict[str, Any]:
    if str(args.variant).upper() != "BAYESADAPTER":
        return {"prior_mu": None, "prior_log_sigma": None, "bridge_info": None}
    if source_payload is None:
        return {"prior_mu": None, "prior_log_sigma": None, "bridge_info": None}

    mu_strategy = str(getattr(args, "bayesadapter_text_only_mu_strategy", "replace")).strip().lower()
    sigma_strategy = str(getattr(args, "bayesadapter_text_only_sigma_strategy", "ignore")).strip().lower()
    cov_mode = str(getattr(args, "bayesadapter_covariance_mode", "paper_scalar")).strip().lower()

    mu_text = source_payload["prior_mu"].detach().float().cpu()
    if mu_strategy == "replace":
        prior_mu = mu_text
    elif mu_strategy == "blend":
        lam = float(getattr(args, "bayesadapter_text_only_mu_blend_lambda", 1.0))
        base_text = _build_base_text_features_for_bridge(ctx, args).detach().float().cpu()
        if base_text.shape != mu_text.shape:
            raise ValueError(f"base_text_features shape mismatch: {tuple(base_text.shape)} vs {tuple(mu_text.shape)}")
        prior_mu = (1.0 - lam) * base_text + lam * mu_text
    else:
        raise ValueError(f"unsupported bayesadapter_text_only_mu_strategy={mu_strategy}")

    if sigma_strategy == "ignore":
        prior_log_sigma = None
    elif sigma_strategy == "override":
        if cov_mode == "diag":
            prior_log_sigma = source_payload["prior_log_sigma_diag"].detach().float().cpu()
        else:
            prior_log_sigma = source_payload["prior_log_sigma_paper"].detach().float().cpu()
    else:
        raise ValueError(f"unsupported bayesadapter_text_only_sigma_strategy={sigma_strategy}")

    bridge_info = {
        "source_run_dir": source_payload["source_run_dir"],
        "source_ckpt": source_payload["source_ckpt"],
        "mu_strategy": mu_strategy,
        "sigma_strategy": sigma_strategy,
        "covariance_mode": cov_mode,
        "lambda_txt": float(source_payload["lambda_txt"]),
        "pseudo_data_count": float(source_payload["pseudo_data_count"]),
        "mean_trace_sigma": float(source_payload["trace_sigma"].mean().item()),
        "mean_scalar_prior_sigma": float(source_payload["prior_log_sigma_paper"].exp().mean().item()),
        "mean_diag_prior_sigma": float(source_payload["prior_log_sigma_diag"].exp().mean().item()),
    }
    if mu_strategy == "blend":
        bridge_info["mu_blend_lambda"] = float(getattr(args, "bayesadapter_text_only_mu_blend_lambda", 1.0))
    return {"prior_mu": prior_mu, "prior_log_sigma": prior_log_sigma, "bridge_info": bridge_info}


@torch.no_grad()
def _build_text_only_bayesadapter_prior(ctx, args) -> dict[str, Any] | None:
    if str(args.variant).upper() != "BAYESADAPTER":
        return None
    if not _has_text_only_bridge(args):
        return None
    run_dir_raw = str(getattr(args, "bayesadapter_text_only_run_dir", "")).strip()
    if not run_dir_raw:
        raise ValueError("bayesadapter_text_only_run_dir must be provided when text-only bridge is enabled")
    run_dir = Path(run_dir_raw).expanduser().resolve()
    config_path = run_dir / "config" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing text_only config/config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        text_only_cfg = json.load(f)
    if str(text_only_cfg.get("family", text_only_cfg.get("recipe_name", ""))) != "text_only_bayes_coop":
        raise ValueError(f"{config_path} does not describe a text_only_bayes_coop run")
    if str(text_only_cfg.get("dataset", "")) != str(args.dataset):
        raise ValueError("text_only run_dir dataset mismatch")
    saved_model = str(text_only_cfg.get("model_str", text_only_cfg.get("model", "")))
    if saved_model and saved_model != str(args.model):
        raise ValueError("text_only run_dir model mismatch")
    saved_class_names = _read_saved_class_names(run_dir)
    if saved_class_names is not None and list(saved_class_names) != list(ctx.class_names):
        raise ValueError("text_only class_names mismatch with current task")

    hessian_dir_path = resolve_existing_path(text_only_cfg["hessian_dir"])
    A_txt, B_txt = load_hessians(str(hessian_dir_path), tag="txt", return_info=False)
    lambda_txt = float(text_only_cfg["lambda_txt"])
    pseudo_data_count = float(text_only_cfg["pseudo_data_count"])
    text_covariance = compute_text_covariance(
        A_txt=A_txt.to(args.device),
        B_txt=B_txt.to(args.device),
        n_txt=pseudo_data_count,
        lambda_txt=lambda_txt,
    )


    prompt_learner, text_only_model = build_text_only_bayes_coop_model(
        class_names=ctx.class_names,
        text_encoder=ctx.text_encoder,
        image_encoder=ctx.image_encoder,
        vlm=ctx.vlm,
        text_covariance=text_covariance,
        n_ctx=int(text_only_cfg["n_ctx"]),
        ctx_init=str(text_only_cfg.get("ctx_init", "")),
        csc=bool(text_only_cfg.get("csc", False)),
        class_token_position=str(text_only_cfg.get("class_token_position", "end")),
        use_full_cov=bool(text_only_cfg.get("use_full_cov", False)),
        train_logit_scale=bool(text_only_cfg.get("train_logit_scale", False)),
        device=args.device,
    )


    ckpt_path = _resolve_text_only_ckpt_path(run_dir, text_only_cfg, str(getattr(args, "bayesadapter_text_only_ckpt", "auto")))
    ckpt_obj = torch.load(ckpt_path, map_location=args.device)
    prompt_state_dict = _extract_prompt_state_dict(ckpt_obj)
    prompt_learner.load_state_dict(prompt_state_dict, strict=True)
    text_only_model.eval()
    mu, _, alpha, trace_sigma = text_only_model.compute_text_statistics()
    mu = mu.detach().float().cpu()
    alpha = alpha.detach().float().cpu()
    trace_sigma = trace_sigma.detach().float().cpu()
    feat_dim = int(mu.shape[1])
    B_inv = text_only_model.text_covariance.B_inv.detach().float().cpu()
    diag_B = torch.diagonal(B_inv, 0)
    scalar_prior_sigma = torch.sqrt((trace_sigma / float(feat_dim)).clamp_min(1e-12))
    prior_log_sigma_paper = scalar_prior_sigma.log()
    diag_prior_var = (alpha[:, None] * diag_B[None, :]).clamp_min(1e-12)
    prior_log_sigma_diag = 0.5 * torch.log(diag_prior_var)
    return {
        "source_run_dir": str(run_dir),
        "source_ckpt": str(ckpt_path),
        "lambda_txt": lambda_txt,
        "pseudo_data_count": pseudo_data_count,
        "prior_mu": mu,
        "prior_log_sigma_paper": prior_log_sigma_paper,
        "prior_log_sigma_diag": prior_log_sigma_diag,
        "alpha": alpha,
        "trace_sigma": trace_sigma,
    }


def _public_namespace_dict(args) -> dict[str, Any]:
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if torch.is_tensor(value):
        return {"_type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)


def _build_model_cfg_from_args(args, resolved_prior: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(_public_namespace_dict(args))
    cfg["adapter_name"] = str(args.variant).upper()
    cfg.setdefault("model_name_or_path", cfg.get("local_model_path"))
    cfg.setdefault("datasetname", cfg.get("dataset"))
    if resolved_prior is not None:
        prior_mu = resolved_prior.get("prior_mu", None)
        prior_log_sigma = resolved_prior.get("prior_log_sigma", None)
        if prior_mu is not None:
            cfg["bayesadapter_prior_mu"] = prior_mu
        else:
            cfg.pop("bayesadapter_prior_mu", None)
        if prior_log_sigma is not None:
            cfg["bayesadapter_prior_log_sigma"] = prior_log_sigma
        else:
            cfg.pop("bayesadapter_prior_log_sigma", None)
    return cfg


class VLMAdapterFamily(BaseFamily):
    family_name = "vlm_adapter"
    best_checkpoint_filename = "best_adapter.pt"
    last_checkpoint_filename = "last_adapter.pt"
    require_image_feature_cache = True

    default_optimizer_name = "adamw"
    default_scheduler_name = "none"
    default_selection_metric = "loss"
    default_selection_mode = "auto"


    def run_path_parts(self, args) -> list[str]:
        parts = [str(args.initialization)]

        train_logit_scale = bool(getattr(args, "train_logit_scale", False))
        parts.append("logit_scale_train" if train_logit_scale else "logit_scale_frozen")

        if str(args.variant).upper() == "BAYESADAPTER":
            cov_mode = str(getattr(args, "bayesadapter_covariance_mode", "paper_scalar")).lower()
            if cov_mode != "paper_scalar":
                parts.append(f"cov_{cov_mode}")
            if _has_text_only_bridge(args):
                source_dir_name = _slugify_path_part(
                    Path(str(getattr(args, "bayesadapter_text_only_run_dir", "")).strip() or "text_only").name
                )
                ckpt_tag = _slugify_path_part(str(getattr(args, "bayesadapter_text_only_ckpt", "auto")))
                parts.append(f"prior_text_only__{source_dir_name}__{ckpt_tag}")

        parts.append(f"shot_{args.shots_per_class}")
        return parts



    def validate_and_note(self, args) -> None:
        if getattr(args, "hessian_dir", None):
            print(f"[note] hessian_dir={args.hessian_dir} is ignored in cached adapter training.")
        print(f"[note] pseudo_data_count={getattr(args, 'pseudo_data_count', None)} retained only for explicit config parity.")
        if str(args.variant).upper() == "BAYESADAPTER":
            print(f"[note] bayesadapter_covariance_mode={getattr(args, 'bayesadapter_covariance_mode', 'paper_scalar')}")
            if _has_text_only_bridge(args):
                run_dir = str(getattr(args, "bayesadapter_text_only_run_dir", "")).strip()
                if not run_dir:
                    raise ValueError("bayesadapter_text_only_run_dir must be provided when text-only bridge is enabled")
                print(f"[note] bayesadapter_text_only_run_dir={run_dir}")
                print(f"[note] bayesadapter_text_only_ckpt={getattr(args, 'bayesadapter_text_only_ckpt', 'auto')}")




    def build_state(self, ctx, args) -> dict[str, Any]:
        text_only_source_payload = _build_text_only_bayesadapter_prior(ctx, args)
        resolved_prior = _resolve_bayesadapter_canonical_prior(ctx=ctx, args=args, source_payload=text_only_source_payload)
        cfg = _build_model_cfg_from_args(args, resolved_prior=resolved_prior)

        model = build_vlm_adapter_model(
            cfg=cfg,
            class_names=ctx.class_names,
            image_encoder=ctx.image_encoder,
            text_encoder=ctx.text_encoder,
            vlm=ctx.vlm,
            device=args.device,
        )

        if str(args.variant).upper() == "TIPA" and hasattr(model.adapter, "init_tipadapter"):
            print("[TipA] initialize cache_keys / cache_values from cached train features")
            all_features, all_labels = [], []
            for batch in ctx.train_eval_loader:
                feats = batch["image_embeds"].to(args.device)
                labels = batch["class_id"].to(args.device)
                all_features.append(feats.detach().cpu())
                all_labels.append(labels.detach().cpu())
            model.adapter.init_tipadapter(torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0))

        zero_shot_test = evaluate_zero_shot_vlm_adapter(
            model=model,
            loader=ctx.test_loader,
            num_classes=len(ctx.class_names),
            device=args.device,
        )

        optimizer = build_optimizer_from_args(
            model.trainable_parameters(),
            args,
            default_name=self.default_optimizer_name,
        )

        return {
            "cfg": cfg,
            "model": model,
            "optimizer": optimizer,
            "zero_shot_test": zero_shot_test,
            "bayesadapter_bridge_info": resolved_prior["bridge_info"],
            "resolved_family_args": _to_jsonable(_public_namespace_dict(args)),
            "train_logit_scale": bool(getattr(args, "train_logit_scale", False)),
        }



    def build_config_extra(self, state, ctx, args):
        del ctx
        extra = {
            "family": self.family_name,
            "variant": args.variant,
            "initialization": args.initialization,
            "resolved_family_args": state["resolved_family_args"],
            "zero_shot_test": state["zero_shot_test"],
            "train_logit_scale": bool(state.get("train_logit_scale", False)),
        }
        bridge_info = state.get("bayesadapter_bridge_info", None)
        if bridge_info is not None:
            extra["bayesadapter_prior_bridge"] = bridge_info
        return extra




    def train_one_epoch(self, state, ctx, args, epoch):
        model = state["model"]
        optimizer = state["optimizer"]
        train_logit_scale = bool(state.get("train_logit_scale", False))

        model.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_reg_sum = 0.0
        epoch_crossmodal_text_sum = 0.0
        epoch_count = 0
        reg_info = {}

        for batch in ctx.train_loader:
            labels = batch["class_id"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch=batch)
            ce_loss = compute_classification_loss_from_logits(logits, labels)
            reg_loss, reg_info = compute_adapter_regularization_loss(model)
            total_loss = ce_loss + reg_loss

            if str(args.variant).upper() == "CROSSMODAL":
                aux_text_loss = compute_crossmodal_text_loss(
                    model=model,
                    batch_size=labels.size(0),
                    device=args.device,
                )
                total_loss = total_loss + aux_text_loss
                epoch_crossmodal_text_sum += aux_text_loss.item() * labels.size(0)

            total_loss.backward()
            optimizer.step()

            if train_logit_scale and hasattr(model, "logit_scale") and model.logit_scale is not None:
                with torch.no_grad():
                    model.logit_scale.clamp_(max=math.log(100.0))

            epoch_loss_sum += total_loss.item() * labels.size(0)
            epoch_reg_sum += reg_loss.item() * labels.size(0)
            epoch_count += labels.size(0)

        row = {
            "train_loss_step_mean": epoch_loss_sum / max(epoch_count, 1),
            "loss_reg": epoch_reg_sum / max(epoch_count, 1),
        }

        for key in ["loss_kl_raw", "loss_kl", "kl_weight"]:
            if key in reg_info:
                row[key] = reg_info[key]

        if str(args.variant).upper() == "CROSSMODAL":
            row["loss_crossmodal_text"] = epoch_crossmodal_text_sum / max(epoch_count, 1)

        return row





    def evaluate_split(self, state, loader, class_names, ctx, args):
        del ctx
        return evaluate_vlm_adapter(model=state["model"], loader=loader, num_classes=len(class_names), device=args.device)

    def collect_predictions(self, state, loader, class_names, ctx, args, split_name, topk):
        del ctx
        return collect_vlm_adapter_predictions(
            model=state["model"],
            loader=loader,
            class_names=class_names,
            device=args.device,
            split_name=split_name,
            topk=topk,
        )

    def collect_ood_payload(self, state, loader, ctx, args):
        del ctx
        model = state["model"]
        model.eval()
        all_labels, all_preds, all_probs, all_logits = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch["class_id"].to(args.device)
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

    def format_epoch_log(self, row, ctx, args):
        del ctx, args
        val_metrics = row["val"]
        lr_part = f"lr={row['lr']:.6f} " if "lr" in row else ""
        log_msg = (
            f"[Epoch {row['epoch']:03d}] {lr_part}"
            f"train_loss={row['train_loss_step_mean']:.4f} "
            f"loss_reg={row['loss_reg']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_nlpd={val_metrics['nlpd']:.4f} "
            f"val_ece={val_metrics['ece']:.4f}"
        )
        if "loss_crossmodal_text" in row:
            log_msg += f" loss_crossmodal_text={row['loss_crossmodal_text']:.4f}"
        if "kl_weight" in row:
            log_msg += f" kl_weight={row['kl_weight']:.6f}"
        if "loss_kl_raw" in row:
            log_msg += f" loss_kl_raw={row['loss_kl_raw']:.4f}"
        return log_msg


    def build_best_state(self, state, ctx, args, epoch, val_metrics):
        del ctx, args
        train_logit_scale = bool(state.get("train_logit_scale", False))
        return {
            "adapter": copy.deepcopy(state["model"].adapter.state_dict()),
            "logit_scale": (
                state["model"].logit_scale.detach().cpu().clone()
                if train_logit_scale and getattr(state["model"], "logit_scale", None) is not None
                else None
            ),
            "best_epoch": epoch,
            "best_val_metrics": val_metrics,
        }


    def load_best_state(self, state, best_state, ctx, args):
        del ctx, args
        state["model"].adapter.load_state_dict(best_state["adapter"])
        if "logit_scale" in best_state and best_state["logit_scale"] is not None:
            state["model"].logit_scale.data.copy_(
                best_state["logit_scale"].to(
                    device=state["model"].logit_scale.device,
                    dtype=state["model"].logit_scale.dtype,
                )
            )



    def build_summary_extra(self, state, selected_state, ctx, args):
        del selected_state, ctx
        extra = {
            "variant": args.variant,
            "initialization": args.initialization,
            "zero_shot_test": state["zero_shot_test"],
            "resolved_family_args": state["resolved_family_args"],
        }
        bridge_info = state.get("bayesadapter_bridge_info", None)
        if bridge_info is not None:
            extra["bayesadapter_prior_bridge"] = bridge_info
        return extra
