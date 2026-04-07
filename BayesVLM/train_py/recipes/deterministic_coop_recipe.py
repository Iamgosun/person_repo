from __future__ import annotations

from typing import Any

import torch

from bayesvlm.methods.deterministic_coop import (
    build_deterministic_coop_model,
    dump_deterministic_coop_predictions,
    evaluate_deterministic_coop,
)
from train_py.recipes.base import BaseRecipe
from train_py.train_runtime import build_optimizer_from_args


class DeterministicCoOpRecipe(BaseRecipe):
    method_name = "deterministic_coop"
    best_checkpoint_filename = "best_prompt_learner.pt"
    last_checkpoint_filename = "last_prompt_learner.pt"
    require_image_feature_cache = False

    # 与旧版独立脚本保持一致：prompt 使用 SGD + cosine，best 按 acc 选。
    default_optimizer_name = "sgd"
    default_scheduler_name = "cosine"
    default_selection_metric = "acc"
    default_selection_mode = "auto"

    def run_path_parts(self, args) -> list[str]:
        return [f"shot_{args.shots_per_class}"]

    def build_state(self, ctx, args) -> dict[str, Any]:
        print("[1] 构建标准 deterministic CoOp 模型 ...")
        prompt_learner, model = build_deterministic_coop_model(
            class_names=ctx.class_names,
            text_encoder=ctx.text_encoder,
            image_encoder=ctx.image_encoder,
            vlm=ctx.vlm,
            n_ctx=args.n_ctx,
            ctx_init=args.ctx_init,
            csc=getattr(args, "csc", False),
            class_token_position=getattr(args, "class_token_position", "end"),
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
        }

    def build_config_extra(self, state: dict[str, Any], ctx, args) -> dict[str, Any]:
        return {
            "n_ctx": args.n_ctx,
            "ctx_init": args.ctx_init,
            "csc": getattr(args, "csc", False),
            "class_token_position": getattr(args, "class_token_position", "end"),
        }

    def train_one_epoch(self, state: dict[str, Any], ctx, args, epoch: int) -> dict[str, Any]:
        model = state["model"]
        optimizer = state["optimizer"]

        model.train()

        epoch_loss_sum = 0.0
        epoch_count = 0

        for batch in ctx.train_loader:
            labels = batch["class_id"].to(args.device)

            optimizer.zero_grad()
            logits = model(batch=batch)
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
                reduction="mean",
            )
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * labels.size(0)
            epoch_count += labels.size(0)

        return {
            "train_loss_step_mean": epoch_loss_sum / max(epoch_count, 1),
        }

    def evaluate_split(self, state: dict[str, Any], loader, ctx, args) -> dict[str, float]:
        return evaluate_deterministic_coop(
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
            f"train_loss={row['train_loss_step_mean']:.4f} "
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
        dump_deterministic_coop_predictions(
            run_dir=ctx.run_dir,
            split_name="train",
            model=state["model"],
            loader=ctx.train_eval_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )
        dump_deterministic_coop_predictions(
            run_dir=ctx.run_dir,
            split_name="val",
            model=state["model"],
            loader=ctx.val_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )
        dump_deterministic_coop_predictions(
            run_dir=ctx.run_dir,
            split_name="test",
            model=state["model"],
            loader=ctx.test_loader,
            class_names=ctx.class_names,
            device=args.device,
            topk=args.prediction_topk,
        )
