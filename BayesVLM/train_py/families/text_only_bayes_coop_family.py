from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from bayesvlm.hessians import load_hessians, optimize_prior_precision
from bayesvlm.methods.text_only_bayes_coop import (
    build_text_only_bayes_coop_model,
    collect_text_only_bayes_coop_predictions,
    compute_text_covariance,
    compute_text_only_bayes_coop_train_losses,
    evaluate_text_only_bayes_coop,
    _prepare_text_only_bayes_eval_cache,
)
from bayesvlm.common import ProbabilisticLogits
from train_py.families.base_family import BaseFamily
from train_py.runtime.common_context import resolve_existing_path
from train_py.train_runtime import build_optimizer_from_args




def _check_txt_hessian_dir(hessian_dir: Path) -> dict:
    required = [hessian_dir / "A_txt_analytic.pt", hessian_dir / "B_txt_analytic.pt"]
    missing = [str(p) for p in required if not p.exists()]
    return {"ok": len(missing) == 0, "missing_required": missing, "dir": str(hessian_dir)}


def _compose_train_loss(*, objective: str, epoch: int, hybrid_warmup_epochs: int, loss_dict: dict[str, torch.Tensor], map_loss_weight: float, bayes_loss_weight: float, ctx_reg_weight: float) -> torch.Tensor:
    map_loss = loss_dict["map_loss"]
    bayes_loss = loss_dict["bayes_loss"]
    ctx_reg = loss_dict["ctx_reg"]
    if objective == "map":
        return map_loss + ctx_reg_weight * ctx_reg
    if objective == "bayes":
        return bayes_loss + ctx_reg_weight * ctx_reg
    if objective == "hybrid":
        if epoch <= hybrid_warmup_epochs:
            return map_loss + ctx_reg_weight * ctx_reg
        return map_loss_weight * map_loss + bayes_loss_weight * bayes_loss + ctx_reg_weight * ctx_reg
    raise ValueError(f"unknown train_objective: {objective}")


class TextOnlyBayesCoOpFamily(BaseFamily):
    family_name = "text_only_bayes_coop"
    best_checkpoint_filename = "best_prompt_learner.pt"
    last_checkpoint_filename = "last_prompt_learner.pt"
    require_image_feature_cache = False

    default_optimizer_name = "sgd"
    default_scheduler_name = "cosine"
    default_selection_metric = "acc"
    default_selection_mode = "auto"


    def run_path_parts(self, args) -> list[str]:
        train_logit_scale = bool(getattr(args, "train_logit_scale", False))
        logit_scale_tag = "logit_scale_train" if train_logit_scale else "logit_scale_frozen"
        return [
            logit_scale_tag,
            f"shot_{args.shots_per_class}",
        ]


    def validate_and_note(self, args) -> None:
        hessian_dir_path = resolve_existing_path(args.hessian_dir)
        hessian_check = _check_txt_hessian_dir(hessian_dir_path)
        if not hessian_check["ok"]:
            raise FileNotFoundError(
                "text_only_bayes_coop requires A_txt_analytic.pt and B_txt_analytic.pt\n"
                f"hessian_dir = {hessian_dir_path}\n"
                f"missing = {hessian_check['missing_required']}"
            )



    def build_state(self, ctx, args) -> dict[str, Any]:
        hessian_dir_path = resolve_existing_path(args.hessian_dir)
        hessian_check = _check_txt_hessian_dir(hessian_dir_path)
        print("[family] loading txt Hessian and optimizing text prior precision ...")
        A_txt, B_txt = load_hessians(str(hessian_dir_path), tag="txt", return_info=False)
        lambda_txt = optimize_prior_precision(
            projection=ctx.text_encoder.text_projection,
            A=A_txt,
            B=B_txt,
            lmbda_init=args.lambda_txt_init,
            n=args.pseudo_data_count,
            lr=1e-2,
            num_steps=args.lambda_opt_steps,
            device=args.device,
            verbose=True,
        ).item()
        text_covariance = compute_text_covariance(
            A_txt=A_txt.to(args.device),
            B_txt=B_txt.to(args.device),
            n_txt=args.pseudo_data_count,
            lambda_txt=lambda_txt,
        )

        train_logit_scale = bool(getattr(args, "train_logit_scale", False))

        prompt_learner, model = build_text_only_bayes_coop_model(
            class_names=ctx.class_names,
            text_encoder=ctx.text_encoder,
            image_encoder=ctx.image_encoder,
            vlm=ctx.vlm,
            text_covariance=text_covariance,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            csc=getattr(args, "csc", False),
            class_token_position=getattr(args, "class_token_position", "end"),
            use_full_cov=args.use_full_cov,
            train_logit_scale=train_logit_scale,
            device=args.device,
        )

        trainable_params = list(prompt_learner.parameters())
        if train_logit_scale:
            trainable_params.append(model.logit_scale)

        optimizer = build_optimizer_from_args(
            trainable_params,
            args,
            default_name=self.default_optimizer_name,
        )

        return {
            "model": model,
            "prompt_learner": prompt_learner,
            "optimizer": optimizer,
            "ctx_anchor": prompt_learner.ctx.detach().clone(),
            "hessian_dir_path": hessian_dir_path,
            "hessian_check": hessian_check,
            "lambda_txt": lambda_txt,
            "train_logit_scale": train_logit_scale,
        }



    def build_config_extra(self, state, ctx, args):
        del ctx
        return {
            "family": self.family_name,
            "variant": args.variant,
            "hessian_dir": str(state["hessian_dir_path"]),
            "hessian_check": state["hessian_check"],
            "pseudo_data_count": args.pseudo_data_count,
            "lambda_txt_init": args.lambda_txt_init,
            "lambda_opt_steps": args.lambda_opt_steps,
            "lambda_txt": state["lambda_txt"],
            "n_ctx": args.n_ctx,
            "ctx_init": args.ctx_init,
            "csc": getattr(args, "csc", False),
            "class_token_position": getattr(args, "class_token_position", "end"),
            "use_full_cov": args.use_full_cov,
            "train_objective": getattr(args, "train_objective", "hybrid"),
            "hybrid_warmup_epochs": getattr(args, "hybrid_warmup_epochs", 5),
            "map_loss_weight": getattr(args, "map_loss_weight", 1.0),
            "bayes_loss_weight": getattr(args, "bayes_loss_weight", 1.0),
            "ctx_reg_weight": getattr(args, "ctx_reg_weight", 1e-4),
            "save_prototype_history": bool(getattr(args, "save_prototype_history", False)),
            "train_logit_scale": bool(state.get("train_logit_scale", False)),
        }



    def train_one_epoch(self, state, ctx, args, epoch):
        model = state["model"]
        prompt_learner = state["prompt_learner"]
        optimizer = state["optimizer"]
        train_logit_scale = bool(state.get("train_logit_scale", False))

        model.train()
        epoch_total_loss = 0.0
        epoch_map_loss = 0.0
        epoch_bayes_loss = 0.0
        epoch_ctx_reg = 0.0
        epoch_count = 0

        for batch in ctx.train_loader:
            labels = batch["class_id"].to(args.device)
            optimizer.zero_grad()
            loss_dict = compute_text_only_bayes_coop_train_losses(
                model=model,
                prompt_learner=prompt_learner,
                batch=batch,
                labels=labels,
                ctx_anchor=state["ctx_anchor"],
                ctx_reg_weight=getattr(args, "ctx_reg_weight", 1e-4),
            )
            loss = _compose_train_loss(
                objective=getattr(args, "train_objective", "hybrid"),
                epoch=epoch,
                hybrid_warmup_epochs=getattr(args, "hybrid_warmup_epochs", 5),
                loss_dict=loss_dict,
                map_loss_weight=getattr(args, "map_loss_weight", 1.0),
                bayes_loss_weight=getattr(args, "bayes_loss_weight", 1.0),
                ctx_reg_weight=getattr(args, "ctx_reg_weight", 1e-4),
            )
            loss.backward()
            optimizer.step()

            if train_logit_scale:
                with torch.no_grad():
                    model.logit_scale.clamp_(max=math.log(100.0))

            n = labels.size(0)
            epoch_total_loss += loss.detach().item() * n
            epoch_map_loss += loss_dict["map_loss"].detach().item() * n
            epoch_bayes_loss += loss_dict["bayes_loss"].detach().item() * n
            epoch_ctx_reg += loss_dict["ctx_reg"].detach().item() * n
            epoch_count += n

        return {
            "train_loss_step_mean": epoch_total_loss / max(epoch_count, 1),
            "train_map_loss_mean": epoch_map_loss / max(epoch_count, 1),
            "train_bayes_loss_mean": epoch_bayes_loss / max(epoch_count, 1),
            "train_ctx_reg_mean": epoch_ctx_reg / max(epoch_count, 1),
        }



    def evaluate_split(self, state, loader, class_names, ctx, args):
        del ctx
        return evaluate_text_only_bayes_coop(
            model=state["model"],
            loader=loader,
            num_classes=len(class_names),
            device=args.device,
        )

    def collect_predictions(self, state, loader, class_names, ctx, args, split_name, topk):
        del ctx
        return collect_text_only_bayes_coop_predictions(
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
        cache = _prepare_text_only_bayes_eval_cache(model)
        all_labels, all_preds, all_probs, all_logits_mean, all_logits_var = [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch["class_id"].to(args.device)
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
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_probs.append(probs.detach().cpu())
                all_logits_mean.append(prob_logits.mean.detach().cpu())
                all_logits_var.append(prob_logits.var.detach().cpu())
        return {
            "labels": torch.cat(all_labels, dim=0),
            "preds": torch.cat(all_preds, dim=0),
            "probs": torch.cat(all_probs, dim=0),
            "logits_mean": torch.cat(all_logits_mean, dim=0),
            "logits_var": torch.cat(all_logits_var, dim=0),
        }

    def format_epoch_log(self, row, ctx, args):
        del ctx, args
        val_metrics = row["val"]
        lr_part = f"lr={row['lr']:.6f} " if "lr" in row else ""
        return (
            f"[Epoch {row['epoch']:03d}] {lr_part}"
            f"train_total={row['train_loss_step_mean']:.4f} "
            f"train_map={row['train_map_loss_mean']:.4f} "
            f"train_bayes={row['train_bayes_loss_mean']:.4f} "
            f"train_ctx_reg={row['train_ctx_reg_mean']:.6f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_nlpd={val_metrics['nlpd']:.4f} "
            f"val_ece={val_metrics['ece']:.4f}"
        )

    def build_best_state(self, state, ctx, args, epoch, val_metrics):
        del ctx, args
        train_logit_scale = bool(state.get("train_logit_scale", False))
        return {
            "prompt_learner": state["prompt_learner"].state_dict(),
            "logit_scale": (
                state["model"].logit_scale.detach().cpu().clone()
                if train_logit_scale
                else None
            ),
            "best_epoch": epoch,
            "best_val_metrics": val_metrics,
        }


    def load_best_state(self, state, best_state, ctx, args):
        del ctx, args
        state["prompt_learner"].load_state_dict(best_state["prompt_learner"])
        if "logit_scale" in best_state and best_state["logit_scale"] is not None:
            state["model"].logit_scale.data.copy_(
                best_state["logit_scale"].to(
                    device=state["model"].logit_scale.device,
                    dtype=state["model"].logit_scale.dtype,
                )
            )