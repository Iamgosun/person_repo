from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bayesvlm.hessians import load_hessians, optimize_prior_precision
from bayesvlm.methods.text_only_bayes_coop import (
    build_text_only_bayes_coop_model,
    compute_text_covariance,
    compute_text_only_bayes_coop_train_losses,
    dump_text_only_bayes_coop_predictions,
    evaluate_text_only_bayes_coop,
)
from train_py.common_experiment import resolve_existing_path
from train_py.recipes.base import BaseRecipe
from train_py.train_runtime import build_optimizer_from_args


def _check_txt_hessian_dir(hessian_dir: Path) -> dict:
    required = [
        hessian_dir / "A_txt_analytic.pt",
        hessian_dir / "B_txt_analytic.pt",
    ]
    missing = [str(p) for p in required if not p.exists()]

    return {
        "ok": len(missing) == 0,
        "missing_required": missing,
        "dir": str(hessian_dir),
    }


def _compose_train_loss(
    *,
    objective: str,
    epoch: int,
    hybrid_warmup_epochs: int,
    loss_dict: dict[str, torch.Tensor],
    map_loss_weight: float,
    bayes_loss_weight: float,
    ctx_reg_weight: float,
) -> torch.Tensor:
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
        return (
            map_loss_weight * map_loss
            + bayes_loss_weight * bayes_loss
            + ctx_reg_weight * ctx_reg
        )
    raise ValueError(f"未知 train_objective: {objective}")


class TextOnlyBayesCoOpRecipe(BaseRecipe):
    method_name = "text_only_bayes_coop"
    best_checkpoint_filename = "best_prompt_learner.pt"
    last_checkpoint_filename = "last_prompt_learner.pt"
    require_image_feature_cache = False

    # 与旧版独立脚本保持一致：prompt 使用 SGD，默认带 cosine 调度，并按 acc 选 best。
    default_optimizer_name = "sgd"
    default_scheduler_name = "cosine"
    default_selection_metric = "acc"
    default_selection_mode = "auto"

    def run_path_parts(self, args) -> list[str]:
        return [f"shot_{args.shots_per_class}"]

    def validate_and_note(self, args) -> None:
        hessian_dir_path = resolve_existing_path(args.hessian_dir)
        hessian_check = _check_txt_hessian_dir(hessian_dir_path)

        if not hessian_check["ok"]:
            existing_files = []
            if hessian_dir_path.exists():
                existing_files = sorted([p.name for p in hessian_dir_path.iterdir()])

            raise FileNotFoundError(
                "当前 text_only_bayes_coop 训练只会读取 txt Hessian。\n"
                f"hessian_dir = {hessian_dir_path}\n"
                f"缺少文件: {hessian_check['missing_required']}\n"
                f"目录现有文件: {existing_files}\n\n"
                "该方法至少需要：\n"
                "  - A_txt_analytic.pt\n"
                "  - B_txt_analytic.pt\n"
            )

    def build_state(self, ctx, args) -> dict[str, Any]:
        hessian_dir_path = resolve_existing_path(args.hessian_dir)
        hessian_check = _check_txt_hessian_dir(hessian_dir_path)

        print("[1] 加载 txt Hessian 并优化文本投影层先验精度 ...")
        print(f"    hessian_dir = {hessian_dir_path}")

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

        print(f"    n_txt      = {args.pseudo_data_count}")
        print(f"    lambda_txt = {lambda_txt:.6f}")

        text_covariance = compute_text_covariance(
            A_txt=A_txt.to(args.device),
            B_txt=B_txt.to(args.device),
            n_txt=args.pseudo_data_count,
            lambda_txt=lambda_txt,
        )

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
            device=args.device,
        )

        optimizer = build_optimizer_from_args(
            prompt_learner.parameters(),
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
        }

    def build_config_extra(self, state: dict[str, Any], ctx, args) -> dict[str, Any]:
        return {
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
        }

    def train_one_epoch(self, state: dict[str, Any], ctx, args, epoch: int) -> dict[str, Any]:
        model = state["model"]
        prompt_learner = state["prompt_learner"]
        optimizer = state["optimizer"]
 
        # 只调用 model.train()：
        # TextOnlyBayesCoOpModel.train() 内部已经负责：
        # 1) prompt_learner.train(mode)
        # 2) image_encoder.eval()
        # 3) prompt_learner.text_encoder.eval()
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

            batch_n = labels.size(0)
            epoch_total_loss += loss.detach().item() * batch_n
            epoch_map_loss += loss_dict["map_loss"].detach().item() * batch_n
            epoch_bayes_loss += loss_dict["bayes_loss"].detach().item() * batch_n
            epoch_ctx_reg += loss_dict["ctx_reg"].detach().item() * batch_n
            epoch_count += batch_n

        return {
            "train_loss_step_mean": epoch_total_loss / max(epoch_count, 1),
            "train_map_loss_mean": epoch_map_loss / max(epoch_count, 1),
            "train_bayes_loss_mean": epoch_bayes_loss / max(epoch_count, 1),
            "train_ctx_reg_mean": epoch_ctx_reg / max(epoch_count, 1),
        }

    def evaluate_split(self, state: dict[str, Any], loader, ctx, args) -> dict[str, float]:
        return evaluate_text_only_bayes_coop(
            model=state["model"],
            loader=loader,
            num_classes=len(ctx.class_names),
            device=args.device,
        )

    def format_epoch_log(self, row: dict[str, Any], ctx, args) -> str:
        val_metrics = row["val"]
        lr_part = f"lr={row['lr']:.6f} " if "lr" in row else ""
        return (
            f"[Epoch {row['epoch']:03d}] "
            f"{lr_part}"
            f"train_total={row['train_loss_step_mean']:.4f} "
            f"train_map={row['train_map_loss_mean']:.4f} "
            f"train_bayes={row['train_bayes_loss_mean']:.4f} "
            f"train_ctx_reg={row['train_ctx_reg_mean']:.6f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_nlpd={val_metrics['nlpd']:.4f} "
            f"val_ece={val_metrics['ece']:.4f}"
        )

    def build_best_state(
        self,
        state: dict[str, Any],
        ctx,
        args,
        epoch: int,
        val_metrics: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "prompt_learner": state["prompt_learner"].state_dict(),
            "best_epoch": epoch,
            "best_val_metrics": val_metrics,
        }

    def load_best_state(self, state: dict[str, Any], best_state: dict[str, Any], ctx, args) -> None:
        state["prompt_learner"].load_state_dict(best_state["prompt_learner"])

    def dump_predictions(self, state: dict[str, Any], ctx, args) -> None:
        dump_text_only_bayes_coop_predictions(
            run_dir=ctx.run_dir,
            split_name="train",
            model=state["model"],
            loader=ctx.train_eval_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )
        dump_text_only_bayes_coop_predictions(
            run_dir=ctx.run_dir,
            split_name="val",
            model=state["model"],
            loader=ctx.val_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )
        dump_text_only_bayes_coop_predictions(
            run_dir=ctx.run_dir,
            split_name="test",
            model=state["model"],
            loader=ctx.test_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )
