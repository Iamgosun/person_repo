from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributions as dists
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassCalibrationError

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.coop_prompt import CoOpPromptLearner
from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.hessians import (
    KroneckerFactorizedCovariance,
    load_hessians,
    optimize_prior_precision,
)
from bayesvlm.text_only_bayes_coop import TextOnlyBayesCoOpModel
from bayesvlm.utils import (
    get_image_size,
    get_model_type_and_size,
    get_transform,
    load_model,
)


SUPPORTED_DATASETS = [
    "flowers102",
    "food101",
    "cifar10",
    "cifar100",
    "imagenet-r",
    "ucf101",
    "sun397",
]


def set_seed(seed: int):
    """
    中文说明：
    固定随机种子，尽量保证 few-shot 抽样与训练可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_dataset(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def get_class_names(ds):
    """
    中文说明：
    沿用你原项目里常见的数据集字段命名约定，尽量自动提取类别名。
    """
    base_ds = unwrap_dataset(ds)

    class_names = getattr(base_ds, "classes", None)
    if class_names is None:
        class_names = getattr(base_ds, "_label_names", None)
    if class_names is None:
        class_names = getattr(base_ds, "label_names", None)
    if class_names is None:
        class_names = getattr(base_ds, "classnames", None)

    if class_names is None:
        raise ValueError("当前数据集无法自动提取类别名，请手动补充。")

    return list(class_names)


def print_class_counts(ds, split_name="train"):
    class_names = get_class_names(ds)
    counter = Counter()

    for i in range(len(ds)):
        sample = ds[i]
        counter[int(sample["class_id"])] += 1

    print(f"===== {split_name} =====")
    print(f"num_classes: {len(class_names)}")
    for class_id in sorted(counter.keys()):
        print(f"{class_id:3d} | {class_names[class_id]:25s} | {counter[class_id]}")

    return class_names, counter


def build_fewshot_subset(ds, shots_per_class: int, seed: int):
    """
    中文说明：
    从训练集按每类 K-shot 抽样。
    如果 shots_per_class <= 0，则直接返回原训练集。
    """
    if shots_per_class <= 0:
        return ds

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    picked = defaultdict(list)
    for idx in indices:
        label = int(ds[idx]["class_id"])
        if len(picked[label]) < shots_per_class:
            picked[label].append(idx)

    final_indices = []
    for label in sorted(picked.keys()):
        final_indices.extend(picked[label])

    final_indices = sorted(final_indices)
    print(f"[few-shot] 训练集从 {len(ds)} 条样本，抽成 {len(final_indices)} 条样本")
    return Subset(ds, final_indices)


def build_loader(ds, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def compute_text_covariance(
    A_txt: torch.Tensor,
    B_txt: torch.Tensor,
    n_txt: float,
    lambda_txt: float,
) -> KroneckerFactorizedCovariance:
    """
    中文说明：
    只计算文本投影层的后验协方差。
    当前方案是 text-only Bayesian，因此这里不再计算图像侧协方差。
    """
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


@torch.no_grad()
def evaluate_model(
    model: TextOnlyBayesCoOpModel,
    loader: DataLoader,
    num_classes: int,
    device: str,
) -> Dict[str, float]:
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0.0

    ece_metric = MulticlassCalibrationError(
        num_classes=num_classes,
        n_bins=20,
        norm="l1",
    ).to(device)

    for batch in loader:
        labels = batch["class_id"].to(device)
        prob_logits = model(batch=batch)
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
    ece = ece_metric(all_probs, all_labels).item()
    loss = total_loss / len(loader.dataset)

    return {
        "acc": acc,
        "nlpd": nlpd,
        "ece": ece,
        "loss": loss,
    }


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main(
    dataset: str,
    hessian_dir: str,
    model_str: str = "clip-base",
    local_model_path: str = "./models/clip-vit-b32",
    data_root: str = "./datasets",
    pseudo_data_count: int = 4,
    lambda_txt_init: float = 300.0,
    lambda_opt_steps: int = 500,
    n_ctx: int = 16,
    ctx_init: str = "a photo of a",
    shots_per_class: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 32,
    num_workers: int = 4,
    use_full_cov: bool = False,
    save_dir: str = "runs/text_only_bayes_coop",
    seed: int = 42,
    device: str = "cuda",
):
    set_seed(seed)

    if model_str not in MODEL_NAME_MAP:
        raise ValueError(f"Invalid model name: {model_str}, must be one of {list(MODEL_NAME_MAP.keys())}")

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Invalid dataset: {dataset}, must be one of {SUPPORTED_DATASETS}")

    model_type, _ = get_model_type_and_size(model_str)
    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)

    factory = DataModuleFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=transform,
        test_transform=transform,
        shuffle_train=True,
        base_path=data_root,
    )
    dm = factory.create(dataset)
    dm.setup()

    train_ds = build_fewshot_subset(dm.train_ds, shots_per_class=shots_per_class, seed=seed)
    val_ds = dm.val_ds if hasattr(dm, "val_ds") and dm.val_ds is not None else dm.test_ds
    test_ds = dm.test_ds

    class_names, _ = print_class_counts(train_ds, split_name="train")
    print_class_counts(test_ds, split_name="test")

    train_loader = build_loader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = build_loader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = build_loader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    image_encoder, text_encoder, vlm = load_model(
        model_str=model_str,
        device=device,
        local_model_path=local_model_path,
    )

    # 中文说明：
    # 当前方案只学习 CoOp 上下文，冻结图像编码器、文本编码器和 logits 头参数。
    image_encoder.freeze_all_layers()
    text_encoder.freeze_all_layers()
    vlm.logit_scale.requires_grad = False
    if hasattr(vlm, "logit_bias"):
        vlm.logit_bias.requires_grad = False

    A_txt, B_txt = load_hessians(hessian_dir, tag="txt", return_info=False)

    print("[1] 优化文本投影层的先验精度 lambda_txt ...")
    lambda_txt = optimize_prior_precision(
        projection=text_encoder.text_projection,
        A=A_txt,
        B=B_txt,
        lmbda_init=lambda_txt_init,
        n=pseudo_data_count,
        lr=1e-2,
        num_steps=lambda_opt_steps,
        device=device,
        verbose=True,
    ).item()

    print(f"    n_txt      = {pseudo_data_count}")
    print(f"    lambda_txt = {lambda_txt:.6f}")

    text_covariance = compute_text_covariance(
        A_txt=A_txt.to(device),
        B_txt=B_txt.to(device),
        n_txt=pseudo_data_count,
        lambda_txt=lambda_txt,
    )

    prompt_learner = CoOpPromptLearner(
        class_names=class_names,
        text_encoder=text_encoder,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
    ).to(device)

    model = TextOnlyBayesCoOpModel(
        image_encoder=image_encoder,
        prompt_learner=prompt_learner,
        text_covariance=text_covariance,
        logit_scale=vlm.logit_scale,
        logit_bias=getattr(vlm, "logit_bias", None),
        use_full_cov=use_full_cov,
    ).to(device)

    optimizer = torch.optim.AdamW(
        prompt_learner.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    run_dir = Path(save_dir) / dataset / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": dataset,
        "hessian_dir": hessian_dir,
        "model_str": model_str,
        "local_model_path": local_model_path,
        "data_root": data_root,
        "pseudo_data_count": pseudo_data_count,
        "lambda_txt_init": lambda_txt_init,
        "lambda_opt_steps": lambda_opt_steps,
        "lambda_txt": lambda_txt,
        "n_ctx": n_ctx,
        "ctx_init": ctx_init,
        "shots_per_class": shots_per_class,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "use_full_cov": use_full_cov,
        "seed": seed,
        "device": device,
        "num_classes": len(class_names),
    }
    save_json(run_dir / "config.json", config)

    print("[2] 开始训练 CoOp 上下文 ...")
    best_val_loss = float("inf")
    best_state = None
    metrics_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        prompt_learner.train()

        epoch_loss = 0.0
        epoch_count = 0

        for batch in train_loader:
            labels = batch["class_id"].to(device)

            optimizer.zero_grad()
            prob_logits = model(batch=batch)
            loss = prob_logits.cross_entropy(
                labels,
                num_samples=0,
                reduction="mean",
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_count += labels.size(0)

        train_metrics = evaluate_model(model, train_loader, len(class_names), device=device)
        val_metrics = evaluate_model(model, val_loader, len(class_names), device=device)
        test_metrics = evaluate_model(model, test_loader, len(class_names), device=device)

        row = {
            "epoch": epoch,
            "train_loss_step_mean": epoch_loss / max(epoch_count, 1),
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        metrics_history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"val_nlpd={val_metrics['nlpd']:.4f} "
            f"val_ece={val_metrics['ece']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "prompt_learner": prompt_learner.state_dict(),
                "config": config,
                "best_epoch": epoch,
                "best_val_metrics": val_metrics,
                "best_test_metrics": test_metrics,
            }
            torch.save(best_state, run_dir / "best_prompt_learner.pt")

        save_json(run_dir / "metrics_history.json", metrics_history)

    print("[3] 训练结束，加载最优 prompt 并做最终测试 ...")
    if best_state is None:
        raise RuntimeError("训练过程中没有保存到 best_state。")

    prompt_learner.load_state_dict(best_state["prompt_learner"])

    final_val_metrics = evaluate_model(model, val_loader, len(class_names), device=device)
    final_test_metrics = evaluate_model(model, test_loader, len(class_names), device=device)

    summary = {
        "best_epoch": best_state["best_epoch"],
        "best_val_metrics_saved": best_state["best_val_metrics"],
        "best_test_metrics_saved": best_state["best_test_metrics"],
        "final_val_metrics_recomputed": final_val_metrics,
        "final_test_metrics_recomputed": final_test_metrics,
    }
    save_json(run_dir / "summary.json", summary)

    print("[done] 最终结果：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--hessian_dir", type=str, default="hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K")
    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--local_model_path", type=str, default="./models/clip-vit-b32")
    parser.add_argument("--data_root", type=str, default="./datasets")

    parser.add_argument("--pseudo_data_count", type=int, default=4)
    parser.add_argument("--lambda_txt_init", type=float, default=300.0)
    parser.add_argument("--lambda_opt_steps", type=int, default=500)

    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--ctx_init", type=str, default="a photo of a")
    parser.add_argument("--shots_per_class", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_full_cov", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="runs/text_only_bayes_coop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        hessian_dir=args.hessian_dir,
        model_str=args.model,
        local_model_path=args.local_model_path,
        data_root=args.data_root,
        pseudo_data_count=args.pseudo_data_count,
        lambda_txt_init=args.lambda_txt_init,
        lambda_opt_steps=args.lambda_opt_steps,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        shots_per_class=args.shots_per_class,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_full_cov=args.use_full_cov,
        save_dir=args.save_dir,
        seed=args.seed,
        device=args.device,
    )
