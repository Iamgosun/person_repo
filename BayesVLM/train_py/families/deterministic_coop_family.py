from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from bayesvlm.methods.deterministic_coop import (
    build_deterministic_coop_model,
    collect_deterministic_coop_predictions,
    evaluate_deterministic_coop,
    _prepare_deterministic_eval_cache,
)
from train_py.families.base_family import BaseFamily
from train_py.train_runtime import build_optimizer_from_args


class DeterministicCoOpFamily(BaseFamily):
    family_name = "deterministic_coop"
    best_checkpoint_filename = "best_prompt_learner.pt"
    last_checkpoint_filename = "last_prompt_learner.pt"
    require_image_feature_cache = False

    default_optimizer_name = "sgd"
    default_scheduler_name = "cosine"
    default_selection_metric = "acc"
    default_selection_mode = "auto"

    def run_path_parts(self, args) -> list[str]:
        return [f"shot_{args.shots_per_class}"]

    def build_state(self, ctx, args) -> dict[str, Any]:
        print("[family] building deterministic CoOp model ...")
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
        return {"model": model, "prompt_learner": prompt_learner, "optimizer": optimizer}

    def build_config_extra(self, state: dict[str, Any], ctx, args) -> dict[str, Any]:
        del state, ctx
        return {
            "family": self.family_name,
            "variant": args.variant,
            "n_ctx": args.n_ctx,
            "ctx_init": args.ctx_init,
            "csc": getattr(args, "csc", False),
            "class_token_position": getattr(args, "class_token_position", "end"),
        }

    def train_one_epoch(self, state: dict[str, Any], ctx, args, epoch: int) -> dict[str, Any]:
        del epoch
        model = state["model"]
        optimizer = state["optimizer"]
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        for batch in ctx.train_loader:
            labels = batch["class_id"].to(args.device)
            optimizer.zero_grad()
            logits = model(batch=batch)
            loss = F.cross_entropy(logits, labels, reduction="mean")
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * labels.size(0)
            epoch_count += labels.size(0)
        return {"train_loss_step_mean": epoch_loss_sum / max(epoch_count, 1)}

    def evaluate_split(self, state: dict[str, Any], loader, class_names: list[str], ctx, args) -> dict[str, float]:
        del ctx
        return evaluate_deterministic_coop(
            model=state["model"],
            loader=loader,
            num_classes=len(class_names),
            device=args.device,
        )

    def collect_predictions(self, state, loader, class_names, ctx, args, split_name, topk):
        del ctx
        return collect_deterministic_coop_predictions(
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
        cache = _prepare_deterministic_eval_cache(model)
        text_features = cache["text_features"]
        all_labels, all_preds, all_probs, all_logits = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch["class_id"].to(args.device)
                g = model.encode_image_batch(batch=batch)
                logits = model.vlm(g, text_features)
                probs = F.softmax(logits, dim=-1)
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

    def format_epoch_log(self, row: dict[str, Any], ctx, args) -> str:
        del ctx, args
        val_metrics = row["val"]
        lr_part = f"lr={row['lr']:.6f} " if "lr" in row else ""
        return (
            f"[Epoch {row['epoch']:03d}] {lr_part}"
            f"train_loss={row['train_loss_step_mean']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_nlpd={val_metrics['nlpd']:.4f} "
            f"val_ece={val_metrics['ece']:.4f}"
        )

    def build_best_state(self, state, ctx, args, epoch, val_metrics):
        del ctx, args
        return {
            "prompt_learner": state["prompt_learner"].state_dict(),
            "best_epoch": epoch,
            "best_val_metrics": val_metrics,
        }

    def load_best_state(self, state, best_state, ctx, args):
        del ctx, args
        state["prompt_learner"].load_state_dict(best_state["prompt_learner"])
