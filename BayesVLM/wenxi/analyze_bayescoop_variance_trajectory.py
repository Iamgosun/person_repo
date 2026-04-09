import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bayesvlm.training.runtime import set_seed
from train_py.common_experiment import build_common_context
from train_py.recipes import build_recipe


def _to_namespace_from_config(cfg: dict, device: str | None = None):
    """
    根据 run_dir/config.json 重建一个足够给 build_common_context + recipe.build_state 使用的 args。
    注意：
    1. 当前 config.json 里保存的是 local_model_path_raw / data_root_raw，而不是训练脚本入参名；
       这里要映射回 local_model_path / data_root。
    2. 这里只重建 text_only_bayes_coop 分析所需字段。
    """
    args = SimpleNamespace()

    # 通用字段
    args.recipe_name = cfg["recipe_name"]
    args.method_name = cfg.get("method_name", cfg["recipe_name"])
    args.dataset = cfg["dataset"]
    args.model = cfg["model_str"]
    args.local_model_path = cfg["local_model_path_raw"]
    args.data_root = cfg["data_root_raw"]

    args.batch_size = int(cfg["batch_size"])
    args.num_workers = int(cfg["num_workers"])
    args.seed = int(cfg["seed"])
    args.device = device or cfg.get("device", "cuda")

    args.disable_cache_image_features = not bool(cfg.get("cache_image_features", False))
    args.cache_image_features = bool(cfg.get("cache_image_features", False))
    args.image_feature_cache_root = cfg["image_feature_cache_root"]
    args.rebuild_image_feature_cache = False

    args.prediction_topk = int(cfg.get("prediction_topk", 5))
    args.save_dir = str(Path(cfg["run_dir"]).parent.parent.parent) if "run_dir" in cfg else "./output"

    # 优化器/训练字段：虽然分析不训练，但 recipe.build_state 会构造 optimizer，所以这些字段要在
    args.lr = float(cfg.get("lr", 2e-3))
    args.weight_decay = float(cfg.get("weight_decay", 0.0))
    args.epochs = int(cfg.get("epochs", 1))
    args.optimizer = cfg.get("optimizer", "sgd")
    args.momentum = float(cfg.get("momentum", 0.9))
    args.nesterov = bool(cfg.get("nesterov", False))
    args.lr_scheduler = cfg.get("lr_scheduler", "cosine")
    args.warmup_epoch = int(cfg.get("warmup_epoch", 0))
    args.warmup_cons_lr = float(cfg.get("warmup_cons_lr", 1e-5))
    args.model_selection = cfg.get("model_selection", "best")
    args.selection_metric = cfg.get("selection_metric", "acc")
    args.selection_mode = cfg.get("selection_mode", "auto")

    # few-shot / prompt 相关
    args.shots_per_class = int(cfg["shots_per_class"])
    args.n_ctx = int(cfg["n_ctx"])
    args.ctx_init = cfg.get("ctx_init", "")
    args.csc = bool(cfg.get("csc", False))
    args.class_token_position = cfg.get("class_token_position", "end")
    args.use_full_cov = bool(cfg.get("use_full_cov", False))

    # text_only_bayes_coop 专属
    args.hessian_dir = cfg["hessian_dir"]
    args.pseudo_data_count = int(cfg["pseudo_data_count"])
    args.lambda_txt_init = float(cfg["lambda_txt_init"])
    args.lambda_opt_steps = int(cfg["lambda_opt_steps"])
    args.train_objective = cfg.get("train_objective", "hybrid")
    args.hybrid_warmup_epochs = int(cfg.get("hybrid_warmup_epochs", 5))
    args.map_loss_weight = float(cfg.get("map_loss_weight", 1.0))
    args.bayes_loss_weight = float(cfg.get("bayes_loss_weight", 1.0))
    args.ctx_reg_weight = float(cfg.get("ctx_reg_weight", 1e-4))

    # 兼容字段
    args.use_data_augmentation = bool(cfg.get("use_data_augmentation", False))
    args.use_augmented_train_cache = bool(cfg.get("use_augmented_train_cache", False))
    args.train_aug_repeats = int(cfg.get("train_aug_repeats", 20))

    return args


def _extract_prompt_state_dict(ckpt_obj: dict) -> dict:
    """
    兼容当前项目保存格式：
    1. 当前 text_only_bayes_coop 的 checkpoint 是一个 dict，
       其中真正的 prompt state_dict 放在 ckpt_obj["prompt_learner"] 里。
    2. 也兼容直接保存 flat state_dict 的情况。
    """
    if isinstance(ckpt_obj, dict) and "prompt_learner" in ckpt_obj and isinstance(ckpt_obj["prompt_learner"], dict):
        return ckpt_obj["prompt_learner"]

    if isinstance(ckpt_obj, dict):
        # 兼容直接存 state_dict 的情况
        has_tensor_value = any(torch.is_tensor(v) for v in ckpt_obj.values())
        if has_tensor_value:
            return ckpt_obj

    raise ValueError("无法从 checkpoint 中解析出 prompt_learner 的 state_dict。")


@torch.no_grad()
def _collect_checkpoint_stats(state, ctx, args, checkpoint_tag: str):
    """
    计算两层统计：

    A. class-level:
       - alpha_c
       - trace_sigma_c
       - ||mu_c||
       - 类原型与同类训练图像均值的 cosine 相似度

    B. sample-level:
       - 训练样本在真实类别上的 Bayes logit var 平均值
       - 训练样本在真实类别上的 MAP logit 平均值
    """
    model = state["model"]
    model.eval()

    mu, text_acts, alpha, trace_sigma = model.compute_text_statistics()
    mu = mu.float()
    alpha = alpha.float()
    trace_sigma = trace_sigma.float()

    mu_norm = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    num_classes = len(ctx.class_names)

    # 先收集训练集 image embedding
    class_embed_buckets = [[] for _ in range(num_classes)]
    class_true_var_buckets = [[] for _ in range(num_classes)]
    class_true_mean_buckets = [[] for _ in range(num_classes)]
    class_alignment_buckets = [[] for _ in range(num_classes)]

    for batch in ctx.train_eval_loader:
        labels = batch["class_id"].to(args.device)

        g = model.encode_image_batch(batch=batch).float()
        g_norm = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        prob_logits = model.forward_bayes_logits(batch=batch)
        map_logits = model.forward_map_logits(batch=batch)

        batch_index = torch.arange(labels.size(0), device=labels.device)
        true_var = prob_logits.var[batch_index, labels]
        true_mean = map_logits[batch_index, labels]

        own_mu = mu_norm[labels]
        own_alignment = (g_norm * own_mu).sum(dim=-1)

        for i in range(labels.size(0)):
            c = int(labels[i].item())
            class_embed_buckets[c].append(g_norm[i].detach().cpu())
            class_true_var_buckets[c].append(float(true_var[i].detach().cpu().item()))
            class_true_mean_buckets[c].append(float(true_mean[i].detach().cpu().item()))
            class_alignment_buckets[c].append(float(own_alignment[i].detach().cpu().item()))

    class_rows = []
    centroid_cos_list = []
    trace_sigma_list = []
    align_list = []
    true_var_list = []

    for c in range(num_classes):
        if len(class_embed_buckets[c]) > 0:
            class_stack = torch.stack(class_embed_buckets[c], dim=0)
            centroid = class_stack.mean(dim=0)
            centroid = centroid / centroid.norm().clamp_min(1e-6)
            centroid_cos = float((centroid * mu_norm[c].detach().cpu()).sum().item())
            sample_alignment_mean = float(np.mean(class_alignment_buckets[c]))
            sample_alignment_std = float(np.std(class_alignment_buckets[c]))
            true_logit_var_mean = float(np.mean(class_true_var_buckets[c]))
            true_logit_var_std = float(np.std(class_true_var_buckets[c]))
            true_logit_mean_mean = float(np.mean(class_true_mean_buckets[c]))
            n_train = int(len(class_embed_buckets[c]))
        else:
            centroid_cos = float("nan")
            sample_alignment_mean = float("nan")
            sample_alignment_std = float("nan")
            true_logit_var_mean = float("nan")
            true_logit_var_std = float("nan")
            true_logit_mean_mean = float("nan")
            n_train = 0

        row = {
            "checkpoint_tag": checkpoint_tag,
            "class_id": c,
            "class_name": ctx.class_names[c],
            "n_train": n_train,
            "alpha": float(alpha[c].detach().cpu().item()),
            "trace_sigma": float(trace_sigma[c].detach().cpu().item()),
            "mu_norm": float(mu[c].norm().detach().cpu().item()),
            "text_act_norm": float(text_acts[c].norm().detach().cpu().item()),
            "centroid_cos": centroid_cos,
            "sample_alignment_mean": sample_alignment_mean,
            "sample_alignment_std": sample_alignment_std,
            "true_logit_var_mean": true_logit_var_mean,
            "true_logit_var_std": true_logit_var_std,
            "true_logit_mean_mean": true_logit_mean_mean,
        }
        class_rows.append(row)

        if not np.isnan(centroid_cos):
            centroid_cos_list.append(centroid_cos)
        if not np.isnan(sample_alignment_mean):
            align_list.append(sample_alignment_mean)
        if not np.isnan(true_logit_var_mean):
            true_var_list.append(true_logit_var_mean)
        trace_sigma_list.append(float(trace_sigma[c].detach().cpu().item()))

    def _safe_corr(x, y):
        if len(x) < 2 or len(y) < 2:
            return float("nan")
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if np.allclose(x.std(), 0) or np.allclose(y.std(), 0):
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    class_level_alignment = [
        r["sample_alignment_mean"]
        for r in class_rows
        if not np.isnan(r["sample_alignment_mean"])
    ]
    class_level_trace_sigma = [
        r["trace_sigma"]
        for r in class_rows
        if not np.isnan(r["sample_alignment_mean"])
    ]
    class_level_true_var = [
        r["true_logit_var_mean"]
        for r in class_rows
        if not np.isnan(r["true_logit_var_mean"])
    ]
    class_level_centroid_cos = [
        r["centroid_cos"]
        for r in class_rows
        if not np.isnan(r["centroid_cos"])
    ]

    summary_row = {
        "checkpoint_tag": checkpoint_tag,
        "mean_trace_sigma": float(np.mean(trace_sigma_list)),
        "std_trace_sigma": float(np.std(trace_sigma_list)),
        "mean_centroid_cos": float(np.mean(centroid_cos_list)) if len(centroid_cos_list) > 0 else float("nan"),
        "mean_sample_alignment": float(np.mean(align_list)) if len(align_list) > 0 else float("nan"),
        "mean_true_logit_var": float(np.mean(true_var_list)) if len(true_var_list) > 0 else float("nan"),
        "corr_trace_sigma_vs_alignment": _safe_corr(class_level_trace_sigma, class_level_alignment),
        "corr_trace_sigma_vs_centroid_cos": _safe_corr(class_level_trace_sigma, class_level_centroid_cos),
        "corr_true_var_vs_alignment": _safe_corr(class_level_true_var, class_level_alignment),
    }

    return class_rows, summary_row


def _write_csv(rows: list[dict], path: Path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="分析 text_only_bayes_coop 中类原型方差与训练适配关系的轨迹脚本")
    parser.add_argument("--run-dir", type=str, required=True, help="某次 text_only_bayes_coop 实验的 run_dir")
    parser.add_argument("--device", type=str, default=None, help="分析设备，默认沿用 config.json 中的 device")
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default="epoch_checkpoints/epoch_*.pt",
        help="相对于 run_dir 的 checkpoint 通配符；若不存在，则退化为 init/best/last 三点分析",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 config.json: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    rebuilt_args = _to_namespace_from_config(cfg, device=args.device)

    # 这一步很重要：为了重建和训练时一致的 few-shot 子集与初始化，需要先设回原 seed
    set_seed(rebuilt_args.seed)

    recipe = build_recipe("text_only_bayes_coop")
    ctx = build_common_context(
        args=rebuilt_args,
        run_dir=run_dir,
        require_image_feature_cache=recipe.require_image_feature_cache,
    )
    state = recipe.build_state(ctx, rebuilt_args)

    all_class_rows = []
    all_summary_rows = []

    # 先分析 init
    init_class_rows, init_summary_row = _collect_checkpoint_stats(
        state=state,
        ctx=ctx,
        args=rebuilt_args,
        checkpoint_tag="init",
    )
    all_class_rows.extend(init_class_rows)
    all_summary_rows.append(init_summary_row)

    # 再分析 epoch 轨迹；如果没有 epoch ckpt，则至少分析 best/last
    ckpt_paths = sorted((run_dir / ".").glob(args.checkpoint_glob))
    if len(ckpt_paths) > 0:
        for ckpt_path in ckpt_paths:
            ckpt_obj = torch.load(ckpt_path, map_location=rebuilt_args.device)
            prompt_sd = _extract_prompt_state_dict(ckpt_obj)
            state["prompt_learner"].load_state_dict(prompt_sd, strict=True)

            tag = ckpt_path.stem
            class_rows, summary_row = _collect_checkpoint_stats(
                state=state,
                ctx=ctx,
                args=rebuilt_args,
                checkpoint_tag=tag,
            )
            all_class_rows.extend(class_rows)
            all_summary_rows.append(summary_row)
    else:
        for name in ["best_prompt_learner.pt", "last_prompt_learner.pt"]:
            ckpt_path = run_dir / name
            if not ckpt_path.exists():
                continue

            ckpt_obj = torch.load(ckpt_path, map_location=rebuilt_args.device)
            prompt_sd = _extract_prompt_state_dict(ckpt_obj)
            state["prompt_learner"].load_state_dict(prompt_sd, strict=True)

            class_rows, summary_row = _collect_checkpoint_stats(
                state=state,
                ctx=ctx,
                args=rebuilt_args,
                checkpoint_tag=name.replace(".pt", ""),
            )
            all_class_rows.extend(class_rows)
            all_summary_rows.append(summary_row)

    out_dir = run_dir / "variance_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(all_class_rows, out_dir / "class_level_trajectory.csv")
    _write_csv(all_summary_rows, out_dir / "checkpoint_summary.csv")

    with open(out_dir / "checkpoint_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary_rows, f, ensure_ascii=False, indent=2)

    print("分析完成。输出文件：")
    print(f"  - {out_dir / 'class_level_trajectory.csv'}")
    print(f"  - {out_dir / 'checkpoint_summary.csv'}")
    print(f"  - {out_dir / 'checkpoint_summary.json'}")

    print("\n建议重点关注这些列：")
    print("  1) class_level_trajectory.csv:")
    print("     - trace_sigma")
    print("     - centroid_cos")
    print("     - sample_alignment_mean")
    print("     - true_logit_var_mean")
    print("  2) checkpoint_summary.csv:")
    print("     - mean_trace_sigma")
    print("     - mean_centroid_cos")
    print("     - mean_true_logit_var")
    print("     - corr_trace_sigma_vs_alignment")
    print("     - corr_true_var_vs_alignment")


if __name__ == "__main__":
    main()