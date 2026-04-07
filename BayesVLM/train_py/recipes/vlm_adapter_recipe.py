from __future__ import annotations

import copy
from typing import Any

import torch

from bayesvlm.methods.vlm_adapter import (
    build_vlm_adapter_model,
    compute_adapter_regularization_loss,
    compute_crossmodal_text_loss,
    dump_vlm_adapter_predictions,
    evaluate_vlm_adapter,
    evaluate_zero_shot_vlm_adapter,
)
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


class VLMAdapterRecipe(BaseRecipe):
    method_name = "vlm_adapter"
    best_checkpoint_filename = "best_adapter.pt"
    last_checkpoint_filename = "last_adapter.pt"
    require_image_feature_cache = True

    # 当前已验证实现默认使用 AdamW，且原通用训练器按 val loss 选择 best。
    default_optimizer_name = "adamw"
    default_scheduler_name = "none"
    default_selection_metric = "loss"
    default_selection_mode = "auto"

    def run_path_parts(self, args) -> list[str]:
        return [
            args.adapter_name.upper(),
            args.initialization,
            f"shot_{args.shots_per_class}",
        ]

    def validate_and_note(self, args) -> None:
        if args.hessian_dir:
            print(f"[note] hessian_dir={args.hessian_dir} 当前 cached adapter 训练不会直接使用。")
        print(f"[note] pseudo_data_count={args.pseudo_data_count} 仅为兼容旧接口保留。")

    def build_state(self, ctx, args) -> dict[str, Any]:
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
        }

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
        }

    def build_config_extra(self, state: dict[str, Any], ctx, args) -> dict[str, Any]:
        return {
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
            "zero_shot_test": state["zero_shot_test"],
        }

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
            ce_loss = torch.nn.functional.cross_entropy(logits, labels)

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
            log_msg += f" kl_weight={row['kl_weight']:.4f}"
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
        return {
            "adapter_name": args.adapter_name,
            "initialization": args.initialization,
            "zero_shot_test": state["zero_shot_test"],
        }
