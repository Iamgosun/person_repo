from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torchmetrics.classification import MulticlassCalibrationError

from bayesvlm.constants import MODEL_NAME_MAP
from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.hessians import compute_covariances, load_hessians, optimize_prior_precision
from bayesvlm.precompute import precompute_image_features
from bayesvlm.semantic_postnet import SemanticPosteriorNetwork
from bayesvlm.text_priors import build_class_gaussian_priors, default_class_to_prompts
from bayesvlm.utils import get_image_size, get_model_type_and_size, get_transform, load_model


# 预训练的 CLIP 图像编码器（Frozen Image Encoder）与一个新构建的 语义后验网络（SemanticPosteriorNetwork） 结合



@torch.no_grad()
def evaluate_id(model: SemanticPosteriorNetwork, loader: torch.utils.data.DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_labels = []
    all_alpha0 = []
    for embeds, labels in loader:
        embeds = embeds.to(device)
        output = model.predict(embeds)
        all_probs.append(output["probs"].cpu())
        all_labels.append(labels.cpu())
        all_alpha0.append(output["alpha0"].cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    alpha0 = torch.cat(all_alpha0, dim=0)
    predictions = probs.argmax(dim=-1)
    acc = (predictions == labels).float().mean().item()
    brier = ((probs - torch.nn.functional.one_hot(labels, num_classes=probs.shape[-1]).float()) ** 2).sum(dim=-1).mean().item()
    nlpd = (-torch.distributions.Categorical(probs).log_prob(labels)).mean().item()
    ece_metric = MulticlassCalibrationError(num_classes=probs.shape[-1], n_bins=20, norm="l1")
    ece = ece_metric(probs, labels).item()
    return {
        "acc": acc,
        "brier": brier,
        "nlpd": nlpd,
        "ece": ece,
        "alpha0_mean": alpha0.mean().item(),
    }


@torch.no_grad()
def evaluate_ood(model: SemanticPosteriorNetwork, id_loader: torch.utils.data.DataLoader, ood_loader: torch.utils.data.DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    id_alpha0, ood_alpha0 = [], []
    for embeds, _ in id_loader:
        id_alpha0.append(model.predict(embeds.to(device))["alpha0"].cpu())
    for embeds, _ in ood_loader:
        ood_alpha0.append(model.predict(embeds.to(device))["alpha0"].cpu())
    id_alpha0 = torch.cat(id_alpha0, dim=0)
    ood_alpha0 = torch.cat(ood_alpha0, dim=0)

    scores = torch.cat([id_alpha0, ood_alpha0], dim=0)
    labels = torch.cat([torch.ones_like(id_alpha0), torch.zeros_like(ood_alpha0)], dim=0)
    order = torch.argsort(scores, descending=True)
    labels = labels[order]
    tp = labels.cumsum(dim=0)
    fp = torch.arange(1, labels.numel() + 1) - tp
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / tp[-1].clamp_min(1)
    aupr = torch.trapz(precision, recall).item()
    return {
        "id_alpha0_mean": id_alpha0.mean().item(),
        "ood_alpha0_mean": ood_alpha0.mean().item(),
        "epistemic_aupr": aupr,
    }


def load_prompt_map(prompt_file: str | None, class_prompts: Sequence[str]) -> Dict[str, List[str]]:
    if prompt_file is None:
        return default_class_to_prompts(class_prompts)

    path = Path(prompt_file)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        if len(data) != len(class_prompts):
            raise ValueError("Prompt list length must match number of classes.")
        return {str(i): list(prompts) for i, prompts in enumerate(data)}

    if isinstance(data, dict):
        normalized: Dict[str, List[str]] = {}
        for index in range(len(class_prompts)):
            key = str(index)
            if key not in data:
                raise KeyError(f"Missing prompts for class index {key}.")
            value = data[key]
            if not isinstance(value, list) or len(value) == 0:
                raise ValueError(f"Prompts for class {key} must be a non-empty list.")
            normalized[key] = list(value)
        return normalized

    raise ValueError("Prompt file must contain either a list-of-lists or a dict keyed by class index strings.")


def make_feature_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main(args: argparse.Namespace) -> None:
    if args.model not in MODEL_NAME_MAP:
        raise ValueError(f"Invalid model {args.model}. Choices: {list(MODEL_NAME_MAP.keys())}")

    model_type, _ = get_model_type_and_size(args.model)
    image_size = get_image_size(args.model)
    transform = get_transform(model_type, image_size)

    factory = DataModuleFactory(
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        train_transform=transform,
        test_transform=transform,
        shuffle_train=True,
    )
    dm = factory.create(args.dataset)
    dm.setup()

    image_encoder, text_encoder, _ = load_model(args.model, args.device)

    A_txt, B_txt = load_hessians(args.hessian_dir, tag="txt", return_info=False)
    info_txt = {
        "n_img": args.pseudo_data_count,
        "n_txt": args.pseudo_data_count,
    }
    info_txt["lambda_img"] = 1.0
    info_txt["lambda_txt"] = optimize_prior_precision(
        text_encoder.text_projection,
        A=A_txt,
        B=B_txt,
        lmbda_init=args.lambda_init,
        n=info_txt["n_txt"],
        lr=args.lambda_lr,
        num_steps=args.lambda_steps,
        device=args.device,
        verbose=True,
    ).item()

    _, cov_txt = compute_covariances(A_txt, B_txt, A_txt, B_txt, info_txt)

    # 预计算图像特征 (Precompute Image Features)
    print("[1/4] Precomputing image features...")
    with torch.no_grad():
        train_outputs, train_labels, _ = precompute_image_features(image_encoder, dm.train_dataloader())
        val_outputs, val_labels, _ = precompute_image_features(image_encoder, dm.val_dataloader())
        test_outputs, test_labels, _ = precompute_image_features(image_encoder, dm.test_dataloader())

    # 阶段二：构建文本先验 (Build Text-Derived Gaussian Class Priors)
    print("[2/4] Building text-derived Gaussian class priors...")
    class_to_prompts = load_prompt_map(args.prompt_file, dm.class_prompts)
    class_priors = build_class_gaussian_priors(
        text_encoder=text_encoder,
        covariance=cov_txt,
        class_to_prompts=class_to_prompts,
        batch_size=args.feature_batch_size,
        device=args.device,
    )

    class_counts = torch.bincount(train_labels, minlength=len(class_priors.class_names)).float()
    
    model = SemanticPosteriorNetwork(
        image_embed_dim=train_outputs.embeds.shape[-1],
        latent_dim=args.latent_dim,
        class_priors=class_priors,
        class_counts=class_counts,
        projector_bias=args.projector_bias,
        flow_layers=args.flow_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        jitter=args.jitter,
        entropy_regularization=args.entropy_regularization,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_feature_loader(train_outputs.embeds, train_labels, args.train_batch_size, shuffle=True)
    val_loader = make_feature_loader(val_outputs.embeds, val_labels, args.train_batch_size, shuffle=False)
    test_loader = make_feature_loader(test_outputs.embeds, test_labels, args.train_batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    print("[3/4] Training semantic posterior network...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_examples = 0
        for embeds, labels in train_loader:
            embeds = embeds.to(args.device)
            labels = labels.to(args.device)
            output = model(embeds, labels=labels)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = embeds.shape[0]
            running_loss += loss.item() * batch_size
            num_examples += batch_size

        train_loss = running_loss / max(num_examples, 1)
        metrics_val = evaluate_id(model, val_loader, args.device)
        if metrics_val["brier"] < best_val:
            best_val = metrics_val["brier"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_acc={metrics_val['acc']:.4f} val_brier={metrics_val['brier']:.4f} "
            f"val_ece={metrics_val['ece']:.4f} val_alpha0={metrics_val['alpha0_mean']:.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    print("[4/4] Final evaluation...")
    test_metrics = evaluate_id(model, test_loader, args.device)
    print(json.dumps({"test": test_metrics}, indent=2))

    if args.ood_dataset:
        dm_ood = factory.create(args.ood_dataset)
        dm_ood.setup()
        with torch.no_grad():
            ood_outputs, ood_labels, _ = precompute_image_features(image_encoder, dm_ood.test_dataloader())
        ood_loader = make_feature_loader(ood_outputs.embeds, ood_labels, args.train_batch_size, shuffle=False)
        ood_metrics = evaluate_ood(model, test_loader, ood_loader, args.device)
        print(json.dumps({"ood": ood_metrics}, indent=2))
    else:
        ood_metrics = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "semantic_postnet.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_priors_mean": class_priors.mean,
            "class_priors_covariance": class_priors.covariance,
            "class_counts": class_counts,
            "class_names": class_priors.class_names,
            "test_metrics": test_metrics,
            "ood_metrics": ood_metrics,
            "config": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLIP-based semantic posterior network.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ood_dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--hessian_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=None, help="JSON file with per-class prompt lists.")
    parser.add_argument("--output_dir", type=str, default="outputs/semantic_postnet")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=128)
    parser.add_argument("--projector_bias", action="store_true")
    parser.add_argument("--pseudo_data_count", type=int, default=4)
    parser.add_argument("--lambda_init", type=float, default=300.0)
    parser.add_argument("--lambda_lr", type=float, default=1e-2)
    parser.add_argument("--lambda_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--entropy_regularization", type=float, default=1e-5)
    parser.add_argument("--jitter", type=float, default=1e-5)
    main(parser.parse_args())
