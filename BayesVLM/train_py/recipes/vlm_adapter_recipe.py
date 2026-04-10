from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from bayesvlm.hessians import load_hessians
from bayesvlm.methods.text_only_bayes_coop import (
    build_text_only_bayes_coop_model,
    compute_text_covariance,
)
from bayesvlm.methods.vlm_adapter import (
    build_vlm_adapter_model,
    compute_adapter_regularization_loss,
    compute_crossmodal_text_loss,
    compute_classification_loss_from_logits,
    dump_vlm_adapter_predictions,
    evaluate_vlm_adapter,
    evaluate_zero_shot_vlm_adapter,
)
from train_py.common_experiment import resolve_existing_path
from train_py.recipes.base import BaseRecipe
from train_py.train_runtime import build_optimizer_from_args


@torch.no_grad()
def _maybe_init_tipa_from_feature_loader(
    model: Any,
    adapter_name: str,
    train_eval_loader,
    device: str,
) -> None:
    if adapter_name.upper() != "TIPA":
        return

    if not hasattr(model.adapter, "init_tipadapter"):
        return

    print("[TipA] 使用缓存后的 image_embeds 初始化 cache_keys / cache_values")

    all_features = []
    all_labels = []

    for batch in train_eval_loader:
        feats = batch["image_embeds"].to(device)
        labels = batch["class_id"].to(device)
        all_features.append(feats.detach().cpu())
        all_labels.append(labels.detach().cpu())

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    model.adapter.init_tipadapter(features, labels)


def _slugify_path_part(text: str) -> str:
    text = str(text).strip()
    if not text:
        return "empty"

    chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def _extract_prompt_state_dict(ckpt_obj: dict[str, Any]) -> dict[str, torch.Tensor]:
    if (
        isinstance(ckpt_obj, dict)
        and "prompt_learner" in ckpt_obj
        and isinstance(ckpt_obj["prompt_learner"], dict)
    ):
        return ckpt_obj["prompt_learner"]

    if isinstance(ckpt_obj, dict):
        if any(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj

    raise ValueError("无法从 checkpoint 中解析出 prompt_learner 的 state_dict。")


def _read_saved_class_names(run_dir: Path) -> list[str] | None:
    class_names_path = run_dir / "class_names.json"
    if not class_names_path.exists():
        return None

    with open(class_names_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "class_names" in data:
        return list(data["class_names"])
    if isinstance(data, list):
        return list(data)
    return None


def _resolve_text_only_ckpt_path(
    run_dir: Path,
    config: dict[str, Any],
    ckpt_spec: str,
) -> Path:
    spec = str(ckpt_spec or "auto").strip()

    if spec in {"", "auto"}:
        selection = str(config.get("model_selection", "last")).strip().lower()
        spec = "best" if selection == "best" else "last"

    if spec == "best":
        ckpt_path = run_dir / "best_prompt_learner.pt"
    elif spec == "last":
        ckpt_path = run_dir / "last_prompt_learner.pt"
    else:
        candidate = Path(spec)
        ckpt_path = candidate if candidate.is_absolute() else (run_dir / candidate)

    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 text_only_bayes_coop checkpoint: {ckpt_path}")
    return ckpt_path


@torch.no_grad()
def _build_text_only_bayesadapter_prior(ctx, args) -> dict[str, Any] | None:
    adapter_key = str(args.adapter_name).upper()
    prior_source = str(getattr(args, "bayesadapter_prior_source", "base_text")).lower()

    if adapter_key not in {"BAYESADAPTER", "BAYES_ADAPTER"}:
        return None
    if prior_source != "text_only_bayes_coop":
        return None

    run_dir_raw = str(getattr(args, "bayesadapter_text_only_run_dir", "")).strip()
    if not run_dir_raw:
        raise ValueError(
            "当 bayesadapter_prior_source=text_only_bayes_coop 时，"
            "必须传 --bayesadapter_text_only_run_dir。"
        )

    run_dir = Path(run_dir_raw).expanduser().resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 text_only_bayes_coop config.json: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        text_only_cfg = json.load(f)

    if str(text_only_cfg.get("recipe_name", "")) != "text_only_bayes_coop":
        raise ValueError(
            f"{config_path} 对应的 recipe_name 不是 text_only_bayes_coop，"
            f"实际为：{text_only_cfg.get('recipe_name')}"
        )

    if str(text_only_cfg.get("dataset", "")) != str(args.dataset):
        raise ValueError(
            "text_only_bayes_coop run_dir 与当前 vlm_adapter 数据集不一致："
            f"text_only={text_only_cfg.get('dataset')} vs current={args.dataset}"
        )

    saved_model = str(text_only_cfg.get("model_str", text_only_cfg.get("model", "")))
    if saved_model and saved_model != str(args.model):
        raise ValueError(
            "text_only_bayes_coop run_dir 与当前 vlm_adapter 模型不一致："
            f"text_only={saved_model} vs current={args.model}"
        )

    saved_class_names = _read_saved_class_names(run_dir)
    if saved_class_names is not None and list(saved_class_names) != list(ctx.class_names):
        raise ValueError(
            "text_only_bayes_coop 的 class_names 与当前任务不一致，"
            "不能安全地把先验直接桥接到 BayesAdapter。"
        )

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
        device=args.device,
    )

    ckpt_path = _resolve_text_only_ckpt_path(
        run_dir=run_dir,
        config=text_only_cfg,
        ckpt_spec=str(getattr(args, "bayesadapter_text_only_ckpt", "auto")),
    )
    ckpt_obj = torch.load(ckpt_path, map_location=args.device)
    prompt_state_dict = _extract_prompt_state_dict(ckpt_obj)
    prompt_learner.load_state_dict(prompt_state_dict, strict=True)

    text_only_model.eval()
    mu, _, alpha, trace_sigma = text_only_model.compute_text_statistics()

    mu = mu.detach().float().cpu()
    alpha = alpha.detach().float().cpu()
    trace_sigma = trace_sigma.detach().float().cpu()

    if mu.ndim != 2:
        raise ValueError(f"重建得到的 mu 形状非法：{tuple(mu.shape)}")
    if mu.shape[0] != len(ctx.class_names):
        raise ValueError(
            "重建得到的类别数与当前任务不一致："
            f"mu.shape[0]={mu.shape[0]} vs len(class_names)={len(ctx.class_names)}"
        )

    feat_dim = int(mu.shape[1])
    if feat_dim <= 0:
        raise ValueError(f"重建得到的特征维度非法：{feat_dim}")

    B_inv = text_only_model.text_covariance.B_inv.detach().float().cpu()
    diag_B = torch.diagonal(B_inv, 0)

    # 论文一致的 paper-scalar 模式：
    # Sigma_c ≈ sigma_c^2 I_D, sigma_c^2 = trace(Sigma_c) / D
    scalar_prior_sigma = torch.sqrt((trace_sigma / float(feat_dim)).clamp_min(1e-12))
    prior_log_sigma_paper = scalar_prior_sigma.log()

    # 扩展 diag 模式：
    # Sigma_c ≈ diag(alpha_c * diag(B_inv))
    diag_prior_var = (alpha[:, None] * diag_B[None, :]).clamp_min(1e-12)
    prior_log_sigma_diag = 0.5 * torch.log(diag_prior_var)

    print("[BayesAdapter] 使用 text_only_bayes_coop 重建先验")
    print(f"  source_run_dir = {run_dir}")
    print(f"  source_ckpt    = {ckpt_path}")
    print(f"  lambda_txt     = {lambda_txt:.6f}")
    print(f"  feat_dim       = {feat_dim}")
    print(f"  mean(trace_sigma)        = {trace_sigma.mean().item():.6f}")
    print(f"  mean(scalar_prior_sigma) = {scalar_prior_sigma.mean().item():.6f}")

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


class VLMAdapterRecipe(BaseRecipe):
    method_name = "vlm_adapter"
    best_checkpoint_filename = "best_adapter.pt"
    last_checkpoint_filename = "last_adapter.pt"
    require_image_feature_cache = True

    default_optimizer_name = "adamw"
    default_scheduler_name = "none"
    default_selection_metric = "loss"
    default_selection_mode = "auto"

    def run_path_parts(self, args) -> list[str]:
        parts = [
            args.adapter_name.upper(),
            args.initialization,
        ]

        if args.adapter_name.upper() in {"BAYESADAPTER", "BAYES_ADAPTER"}:
            cov_mode = str(getattr(args, "bayesadapter_covariance_mode", "paper_scalar")).lower()
            prior_source = str(getattr(args, "bayesadapter_prior_source", "base_text")).lower()

            # 默认 paper_scalar + base_text 保持老路径，不影响原逻辑
            if cov_mode != "paper_scalar":
                parts.append(f"cov_{cov_mode}")

            if prior_source != "base_text":
                source_dir_name = _slugify_path_part(
                    Path(str(getattr(args, "bayesadapter_text_only_run_dir", "")).strip() or "text_only").name
                )
                ckpt_tag = _slugify_path_part(
                    str(getattr(args, "bayesadapter_text_only_ckpt", "auto"))
                )
                parts.append(f"prior_{prior_source}__{source_dir_name}__{ckpt_tag}")

        parts.append(f"shot_{args.shots_per_class}")
        return parts

    def validate_and_note(self, args) -> None:
        if args.hessian_dir:
            print(f"[note] hessian_dir={args.hessian_dir} 当前 cached adapter 训练不会直接使用。")
        print(f"[note] pseudo_data_count={args.pseudo_data_count} 仅为兼容旧接口保留。")

        adapter_key = str(args.adapter_name).upper()
        if adapter_key in {"BAYESADAPTER", "BAYES_ADAPTER"}:
            cov_mode = str(getattr(args, "bayesadapter_covariance_mode", "paper_scalar")).lower()
            prior_source = str(getattr(args, "bayesadapter_prior_source", "base_text")).lower()

            print(f"[note] bayesadapter_covariance_mode={cov_mode}")
            print(f"[note] bayesadapter_prior_source={prior_source}")

            if prior_source == "text_only_bayes_coop":
                run_dir = str(getattr(args, "bayesadapter_text_only_run_dir", "")).strip()
                if not run_dir:
                    raise ValueError(
                        "当 adapter_name=BAYESADAPTER 且 bayesadapter_prior_source=text_only_bayes_coop 时，"
                        "必须传 --bayesadapter_text_only_run_dir。"
                    )

                print(f"[note] bayesadapter_text_only_run_dir={run_dir}")
                print(f"[note] bayesadapter_text_only_ckpt={args.bayesadapter_text_only_ckpt}")

    def build_state(self, ctx, args) -> dict[str, Any]:
        bayesadapter_prior_payload = _build_text_only_bayesadapter_prior(ctx, args)

        cfg = {
            "model": args.model,
            "model_name_or_path": args.local_model_path,
            "datasetname": args.dataset,
            "adapter_name": args.adapter_name,
            "initialization": args.initialization,
            "device": args.device,
            "epochs": args.epochs,
            "taskres_alpha": args.taskres_alpha,
            "clipa_ratio": args.clipa_ratio,
            "clipa_hidden_dim": None if args.clipa_hidden_dim <= 0 else args.clipa_hidden_dim,
            "tipa_alpha": args.tipa_alpha,
            "tipa_beta": args.tipa_beta,
            "gaussian_prior_sigma": args.gaussian_prior_sigma,
            "gaussian_mc_samples": args.gaussian_mc_samples,
            "gaussian_anneal_start_epoch": args.gaussian_anneal_start_epoch,

            "bayesadapter_prior_sigma": args.bayesadapter_prior_sigma,
            "bayesadapter_train_mc_samples": args.bayesadapter_train_mc_samples,
            "bayesadapter_eval_mc_samples": args.bayesadapter_eval_mc_samples,
            "bayesadapter_kl_scale_divisor": args.bayesadapter_kl_scale_divisor,
            "bayesadapter_covariance_mode": args.bayesadapter_covariance_mode,
            "bayesadapter_prior_source": args.bayesadapter_prior_source,
            "bayesadapter_text_only_run_dir": args.bayesadapter_text_only_run_dir,
            "bayesadapter_text_only_ckpt": args.bayesadapter_text_only_ckpt,

            # 只放给模型用，不进 config.json
            "bayesadapter_prior_mu": None,
            "bayesadapter_prior_log_sigma": None,
        }

        if bayesadapter_prior_payload is not None:
            cfg["bayesadapter_prior_mu"] = bayesadapter_prior_payload["prior_mu"]

            cov_mode = str(args.bayesadapter_covariance_mode).lower()
            if cov_mode == "diag":
                cfg["bayesadapter_prior_log_sigma"] = bayesadapter_prior_payload["prior_log_sigma_diag"]
            else:
                cfg["bayesadapter_prior_log_sigma"] = bayesadapter_prior_payload["prior_log_sigma_paper"]

        model = build_vlm_adapter_model(
            cfg=cfg,
            class_names=ctx.class_names,
            image_encoder=ctx.image_encoder,
            text_encoder=ctx.text_encoder,
            vlm=ctx.vlm,
            device=args.device,
        )

        _maybe_init_tipa_from_feature_loader(
            model=model,
            adapter_name=args.adapter_name,
            train_eval_loader=ctx.train_eval_loader,
            device=args.device,
        )

        zero_shot_test = evaluate_zero_shot_vlm_adapter(
            model=model,
            loader=ctx.test_loader,
            num_classes=len(ctx.class_names),
            device=args.device,
        )
        print(
            f"[zero-shot] "
            f"test_acc={zero_shot_test['acc']:.4f} "
            f"test_nlpd={zero_shot_test['nlpd']:.4f} "
            f"test_ece={zero_shot_test['ece']:.4f}"
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
            "bayesadapter_prior_payload": bayesadapter_prior_payload,
        }

    def build_config_extra(self, state: dict[str, Any], ctx, args) -> dict[str, Any]:
        extra = {
            "adapter_name": args.adapter_name,
            "initialization": args.initialization,
            "hessian_dir_ignored": args.hessian_dir,
            "pseudo_data_count_ignored": args.pseudo_data_count,
            "taskres_alpha": args.taskres_alpha,
            "clipa_ratio": args.clipa_ratio,
            "clipa_hidden_dim": None if args.clipa_hidden_dim <= 0 else args.clipa_hidden_dim,
            "tipa_alpha": args.tipa_alpha,
            "tipa_beta": args.tipa_beta,
            "gaussian_prior_sigma": args.gaussian_prior_sigma,
            "gaussian_mc_samples": args.gaussian_mc_samples,
            "gaussian_anneal_start_epoch": args.gaussian_anneal_start_epoch,

            "bayesadapter_prior_sigma": args.bayesadapter_prior_sigma,
            "bayesadapter_train_mc_samples": args.bayesadapter_train_mc_samples,
            "bayesadapter_eval_mc_samples": args.bayesadapter_eval_mc_samples,
            "bayesadapter_kl_scale_divisor": args.bayesadapter_kl_scale_divisor,
            "bayesadapter_covariance_mode": args.bayesadapter_covariance_mode,
            "bayesadapter_prior_source": args.bayesadapter_prior_source,
            "bayesadapter_text_only_run_dir": args.bayesadapter_text_only_run_dir or None,
            "bayesadapter_text_only_ckpt": args.bayesadapter_text_only_ckpt,

            "zero_shot_test": state["zero_shot_test"],
        }

        prior_payload = state.get("bayesadapter_prior_payload", None)
        if prior_payload is not None:
            extra["bayesadapter_prior_bridge"] = {
                "source_run_dir": prior_payload["source_run_dir"],
                "source_ckpt": prior_payload["source_ckpt"],
                "lambda_txt": float(prior_payload["lambda_txt"]),
                "pseudo_data_count": float(prior_payload["pseudo_data_count"]),
                "mean_trace_sigma": float(prior_payload["trace_sigma"].mean().item()),
                "std_trace_sigma": float(prior_payload["trace_sigma"].std(unbiased=False).item()),
                "mean_scalar_prior_sigma": float(prior_payload["prior_log_sigma_paper"].exp().mean().item()),
                "std_scalar_prior_sigma": float(prior_payload["prior_log_sigma_paper"].exp().std(unbiased=False).item()),
                "mean_diag_prior_sigma": float(prior_payload["prior_log_sigma_diag"].exp().mean().item()),
                "std_diag_prior_sigma": float(prior_payload["prior_log_sigma_diag"].exp().std(unbiased=False).item()),
            }

        return extra

    def train_one_epoch(self, state: dict[str, Any], ctx, args, epoch: int) -> dict[str, Any]:
        model = state["model"]
        optimizer = state["optimizer"]

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

            if args.adapter_name.upper() == "CROSSMODAL":
                aux_text_loss = compute_crossmodal_text_loss(
                    model=model,
                    batch_size=labels.size(0),
                    device=args.device,
                )
                total_loss = total_loss + aux_text_loss
                epoch_crossmodal_text_sum += aux_text_loss.item() * labels.size(0)

            total_loss.backward()
            optimizer.step()

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

        if args.adapter_name.upper() == "CROSSMODAL":
            row["loss_crossmodal_text"] = epoch_crossmodal_text_sum / max(epoch_count, 1)

        return row

    def evaluate_split(self, state: dict[str, Any], loader, ctx, args) -> dict[str, float]:
        return evaluate_vlm_adapter(
            model=state["model"],
            loader=loader,
            num_classes=len(ctx.class_names),
            device=args.device,
        )

    def format_epoch_log(self, row: dict[str, Any], ctx, args) -> str:
        val_metrics = row["val"]
        lr_part = f"lr={row['lr']:.6f} " if "lr" in row else ""

        log_msg = (
            f"[Epoch {row['epoch']:03d}] "
            f"{lr_part}"
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

    def build_best_state(
        self,
        state: dict[str, Any],
        ctx,
        args,
        epoch: int,
        val_metrics: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "adapter": copy.deepcopy(state["model"].adapter.state_dict()),
            "best_epoch": epoch,
            "best_val_metrics": val_metrics,
        }

    def load_best_state(self, state: dict[str, Any], best_state: dict[str, Any], ctx, args) -> None:
        state["model"].adapter.load_state_dict(best_state["adapter"])

    def dump_predictions(self, state: dict[str, Any], ctx, args) -> None:
        dump_vlm_adapter_predictions(
            run_dir=ctx.run_dir,
            split_name="train",
            model=state["model"],
            loader=ctx.train_eval_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )
        dump_vlm_adapter_predictions(
            run_dir=ctx.run_dir,
            split_name="val",
            model=state["model"],
            loader=ctx.val_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )
        dump_vlm_adapter_predictions(
            run_dir=ctx.run_dir,
            split_name="test",
            model=state["model"],
            loader=ctx.test_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )

    def build_summary_extra(
        self,
        state: dict[str, Any],
        best_state: dict[str, Any],
        ctx,
        args,
    ) -> dict[str, Any]:
        extra = {
            "adapter_name": args.adapter_name,
            "initialization": args.initialization,
            "zero_shot_test": state["zero_shot_test"],
            "bayesadapter_covariance_mode": getattr(args, "bayesadapter_covariance_mode", "paper_scalar"),
            "bayesadapter_prior_source": getattr(args, "bayesadapter_prior_source", "base_text"),
        }

        prior_payload = state.get("bayesadapter_prior_payload", None)
        if prior_payload is not None:
            extra["bayesadapter_prior_bridge"] = {
                "source_run_dir": prior_payload["source_run_dir"],
                "source_ckpt": prior_payload["source_ckpt"],
                "lambda_txt": float(prior_payload["lambda_txt"]),
                "mean_trace_sigma": float(prior_payload["trace_sigma"].mean().item()),
                "mean_scalar_prior_sigma": float(prior_payload["prior_log_sigma_paper"].exp().mean().item()),
                "mean_diag_prior_sigma": float(prior_payload["prior_log_sigma_diag"].exp().mean().item()),
            }

        return extra