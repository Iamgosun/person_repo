from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from bayesvlm.hessians import (
    KroneckerFactorizedCovariance,
    load_hessians,
    optimize_prior_precision,
)
from bayesvlm.utils import load_model


@dataclass
class PromptCovStats:
    prompt: str
    alpha: float
    trace_sigma: float
    logdet_sigma: float
    fro_sigma: float
    mu_norm: float


def read_prompt_file(path: str | Path) -> list[str]:
    """
    支持：
    1) txt: 每行一个 prompt
    2) json: ["prompt1", "prompt2", ...]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"JSON prompt file must be a list[str], got: {type(data)}")
        prompts = [str(x).strip() for x in data if str(x).strip()]
    else:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if len(prompts) == 0:
        raise ValueError(f"No prompts found in: {path}")

    return prompts


def _append_bias_if_needed(activations: torch.Tensor, projection: torch.nn.Module) -> torch.Tensor:
    has_bias = getattr(projection, "bias", None) is not None
    if not has_bias:
        return activations

    ones = torch.ones(
        activations.shape[0],
        1,
        device=activations.device,
        dtype=activations.dtype,
    )
    return torch.cat([activations, ones], dim=-1)


def build_text_covariance(
    *,
    text_encoder: Any,
    hessian_dir: str,
    pseudo_data_count: int,
    lambda_txt_init: float,
    lambda_opt_steps: int,
    device: str,
    use_saved_prior: bool,
) -> tuple[KroneckerFactorizedCovariance, float, float]:
    """
    构造文本侧 Kronecker 后验协方差。

    注意：
    - 这里不能用 @torch.no_grad()
    - 因为 optimize_prior_precision() 内部要 backward()
    """
    if use_saved_prior:
        A_txt, B_txt, info = load_hessians(hessian_dir, tag="txt", return_info=True)
        n_txt = float(info["n_txt"])
        lambda_txt = float(info["lambda_txt"])
    else:
        A_txt, B_txt = load_hessians(hessian_dir, tag="txt", return_info=False)
        n_txt = float(pseudo_data_count)

        lambda_txt = optimize_prior_precision(
            projection=text_encoder.text_projection,
            A=A_txt,
            B=B_txt,
            lmbda_init=lambda_txt_init,
            n=n_txt,
            lr=1e-2,
            num_steps=lambda_opt_steps,
            device=device,
            verbose=True,
        ).item()

    A_txt = A_txt.to(device).float()
    B_txt = B_txt.to(device).float()

    sqrt_n = math.sqrt(float(n_txt))
    sqrt_lambda = math.sqrt(float(lambda_txt))

    A_post = A_txt * sqrt_n + sqrt_lambda * torch.eye(
        A_txt.shape[0], device=device, dtype=A_txt.dtype
    )
    B_post = B_txt * sqrt_n + sqrt_lambda * torch.eye(
        B_txt.shape[0], device=device, dtype=B_txt.dtype
    )

    text_cov = KroneckerFactorizedCovariance(
        A_inv=torch.linalg.inv(A_post),
        B_inv=torch.linalg.inv(B_post),
    )
    return text_cov, float(lambda_txt), float(n_txt)


@torch.no_grad()
def encode_prompts(
    *,
    text_encoder: Any,
    prompts: Sequence[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回：
        mu:   [N, D]
        acts: [N, D_txt or D_txt+1]
    """
    outputs = text_encoder(list(prompts), return_activations=True)

    # EncoderResult
    mu = outputs.embeds.to(device).float()
    acts = outputs.activations.to(device).float()
    acts = _append_bias_if_needed(acts, text_encoder.text_projection)

    return mu, acts


@torch.no_grad()
def compute_prompt_cov_stats(
    *,
    text_encoder: Any,
    text_covariance: KroneckerFactorizedCovariance,
    prompts: Sequence[str],
    device: str,
    jitter: float = 1e-6,
) -> tuple[list[PromptCovStats], dict[str, torch.Tensor]]:
    mu, acts = encode_prompts(
        text_encoder=text_encoder,
        prompts=prompts,
        device=device,
    )

    A_inv = text_covariance.A_inv.to(device).float()
    B_inv = text_covariance.B_inv.to(device).float()

    # alpha[n] = f_n^T A_inv f_n
    alpha = torch.einsum("bi,ij,bj->b", acts, A_inv, acts).clamp_min(0.0)

    # cov_n = alpha_n * B_inv
    trace_sigma = alpha * torch.trace(B_inv)
    mu_norm = mu.norm(dim=-1)

    eye = torch.eye(B_inv.shape[0], device=device, dtype=B_inv.dtype)

    rows: list[PromptCovStats] = []
    for i, prompt in enumerate(prompts):
        cov_i = alpha[i] * B_inv + jitter * eye
        cov_i = cov_i.double()

        sign, logdet = torch.linalg.slogdet(cov_i)
        logdet_val = float(logdet.item()) if sign > 0 else float("nan")
        fro_val = float(torch.linalg.matrix_norm(cov_i, ord="fro").item())

        rows.append(
            PromptCovStats(
                prompt=str(prompt),
                alpha=float(alpha[i].item()),
                trace_sigma=float(trace_sigma[i].item()),
                logdet_sigma=logdet_val,
                fro_sigma=fro_val,
                mu_norm=float(mu_norm[i].item()),
            )
        )

    payload = {
        "prompts": list(prompts),
        "mu": mu.detach().cpu(),
        "acts": acts.detach().cpu(),
        "alpha": alpha.detach().cpu(),
        "trace_sigma": trace_sigma.detach().cpu(),
        "B_inv": B_inv.detach().cpu(),
    }
    return rows, payload


def permutation_test_mean_diff(
    x: torch.Tensor,
    y: torch.Tensor,
    num_perm: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """
    两独立样本的均值差异 permutation test
    """
    x = x.flatten().float().cpu()
    y = y.flatten().float().cpu()

    obs = (x.mean() - y.mean()).abs()
    pooled = torch.cat([x, y], dim=0)
    n_x = x.numel()

    g = torch.Generator()
    g.manual_seed(seed)

    count = 0
    for _ in range(num_perm):
        perm = pooled[torch.randperm(pooled.numel(), generator=g)]
        x_perm = perm[:n_x]
        y_perm = perm[n_x:]
        stat = (x_perm.mean() - y_perm.mean()).abs()
        if stat >= obs:
            count += 1

    p_value = (count + 1) / (num_perm + 1)
    return {
        "obs_abs_mean_diff": float(obs.item()),
        "p_value": float(p_value),
    }


def paired_sign_flip_test(
    x: torch.Tensor,
    y: torch.Tensor,
    num_perm: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """
    配对样本均值差异检验。
    适合“同一批图像下，A 组 prompt vs B 组 prompt”的 predictive variance 比较。
    """
    x = x.flatten().float().cpu()
    y = y.flatten().float().cpu()

    if x.numel() != y.numel():
        raise ValueError(f"paired_sign_flip_test requires same length, got {x.numel()} vs {y.numel()}")

    diff = x - y
    obs = diff.mean().abs()

    g = torch.Generator()
    g.manual_seed(seed)

    count = 0
    for _ in range(num_perm):
        signs = torch.randint(0, 2, diff.shape, generator=g, dtype=torch.int64)
        signs = signs.float() * 2.0 - 1.0
        stat = (diff * signs).mean().abs()
        if stat >= obs:
            count += 1

    p_value = (count + 1) / (num_perm + 1)
    return {
        "obs_abs_mean_diff": float(obs.item()),
        "p_value": float(p_value),
    }


def _tensor_from_loaded_object(obj: Any) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj

    # EncoderResult or类似对象
    if hasattr(obj, "embeds") and torch.is_tensor(obj.embeds):
        return obj.embeds

    # dict 形式
    if isinstance(obj, dict):
        for key in ["embeds", "image_embeds", "features"]:
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key]

    raise ValueError(
        "Unsupported image embeddings file format. "
        "Expected a Tensor, an object with .embeds, or a dict containing "
        "'embeds' / 'image_embeds' / 'features'."
    )


def load_image_embeddings(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    tensor = _tensor_from_loaded_object(obj)
    if tensor.ndim != 2:
        raise ValueError(f"Expected image embeddings with shape [N, D], got {tuple(tensor.shape)}")
    return tensor.float()


@torch.no_grad()
def compute_predictive_logit_variance(
    *,
    image_embeds: torch.Tensor,
    prompt_payload: dict[str, torch.Tensor],
    logit_scale: torch.Tensor,
    use_full_cov: bool,
) -> torch.Tensor:
    """
    对应 TextOnlyBayesCoOpModel.forward_bayes_logits() 里的 var 部分。

    返回：
        logits_var: [B, C]
    """
    g = image_embeds.float()
    mu = prompt_payload["mu"].to(g.device).float()                     # [C, D]
    alpha = prompt_payload["alpha"].to(g.device).float()              # [C]
    trace_sigma = prompt_payload["trace_sigma"].to(g.device).float()  # [C]
    B_inv = prompt_payload["B_inv"].to(g.device).float()              # [D, D]

    g_norm2 = (g ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    mu_norm2 = (mu ** 2).sum(dim=-1).clamp_min(1e-6)

    if use_full_cov:
        g_quad = torch.einsum("bi,ij,bj->b", g, B_inv, g).unsqueeze(-1)
    else:
        diag_B = torch.diagonal(B_inv)
        g_quad = ((g ** 2) * diag_B.unsqueeze(0)).sum(dim=-1, keepdim=True)

    denom_var = g_norm2 * (mu_norm2 + trace_sigma).unsqueeze(0) + 1e-6
    var_cos = (g_quad * alpha.unsqueeze(0)) / denom_var
    var_cos = var_cos.clamp_min(0.0)

    scale = torch.exp(logit_scale.detach().to(g.device).float())
    logits_var = var_cos * (scale ** 2)
    return logits_var


def summarize_rows(rows: list[PromptCovStats], name: str) -> dict[str, float]:
    alpha_vals = torch.tensor([r.alpha for r in rows], dtype=torch.float32)
    trace_vals = torch.tensor([r.trace_sigma for r in rows], dtype=torch.float32)
    logdet_vals = torch.tensor([r.logdet_sigma for r in rows], dtype=torch.float32)
    fro_vals = torch.tensor([r.fro_sigma for r in rows], dtype=torch.float32)

    summary = {
        "count": len(rows),
        "alpha_mean": float(alpha_vals.mean().item()),
        "alpha_std": float(alpha_vals.std(unbiased=False).item()),
        "trace_sigma_mean": float(trace_vals.mean().item()),
        "trace_sigma_std": float(trace_vals.std(unbiased=False).item()),
        "logdet_sigma_mean": float(torch.nanmean(logdet_vals).item()),
        "fro_sigma_mean": float(fro_vals.mean().item()),
    }

    print(f"\n===== {name} =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    for r in rows:
        print(
            f"[prompt] {r.prompt}\n"
            f"  alpha={r.alpha:.6f} "
            f"trace={r.trace_sigma:.6f} "
            f"logdet={r.logdet_sigma:.6f} "
            f"fro={r.fro_sigma:.6f} "
            f"mu_norm={r.mu_norm:.6f}"
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze BayesVLM text posterior covariance differences between two prompt groups."
    )
    parser.add_argument("--group_a_file", type=str, required=True, help="txt/json file for group A prompts")
    parser.add_argument("--group_b_file", type=str, required=True, help="txt/json file for group B prompts")
    parser.add_argument("--group_a_name", type=str, default="group_a")
    parser.add_argument("--group_b_name", type=str, default="group_b")

    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--local_model_path", type=str, default="./models/clip-vit-b32")
    parser.add_argument("--hessian_dir", type=str, default="./hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--pseudo_data_count", type=int, default=4)
    parser.add_argument("--lambda_txt_init", type=float, default=300.0)
    parser.add_argument("--lambda_opt_steps", type=int, default=1000)
    parser.add_argument(
        "--use_saved_prior",
        action="store_true",
        help="Use n_txt/lambda_txt from prior_precision_analytic.json instead of re-optimizing lambda_txt.",
    )

    parser.add_argument(
        "--image_embeds_path",
        type=str,
        default=None,
        help="Optional .pt file containing real image embeddings [B, D] for predictive variance comparison.",
    )
    parser.add_argument(
        "--use_full_cov",
        action="store_true",
        help="Use g^T B_inv g instead of diagonal approximation when computing predictive logit variance.",
    )

    parser.add_argument("--num_perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="prompt_covariance_report.json")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)

    prompts_a = read_prompt_file(args.group_a_file)
    prompts_b = read_prompt_file(args.group_b_file)

    print(f"[info] device={args.device}")
    print(f"[info] group_a={args.group_a_name}, n={len(prompts_a)}")
    print(f"[info] group_b={args.group_b_name}, n={len(prompts_b)}")

    _, text_encoder, vlm = load_model(
        model_str=args.model,
        device=args.device,
        local_model_path=args.local_model_path,
    )

    text_covariance, lambda_txt, n_txt = build_text_covariance(
        text_encoder=text_encoder,
        hessian_dir=args.hessian_dir,
        pseudo_data_count=args.pseudo_data_count,
        lambda_txt_init=args.lambda_txt_init,
        lambda_opt_steps=args.lambda_opt_steps,
        device=args.device,
        use_saved_prior=args.use_saved_prior,
    )

    print(f"[info] n_txt={n_txt}")
    print(f"[info] lambda_txt={lambda_txt:.6f}")

    rows_a, payload_a = compute_prompt_cov_stats(
        text_encoder=text_encoder,
        text_covariance=text_covariance,
        prompts=prompts_a,
        device=args.device,
    )
    rows_b, payload_b = compute_prompt_cov_stats(
        text_encoder=text_encoder,
        text_covariance=text_covariance,
        prompts=prompts_b,
        device=args.device,
    )

    summary_a = summarize_rows(rows_a, args.group_a_name)
    summary_b = summarize_rows(rows_b, args.group_b_name)

    alpha_a = torch.tensor([r.alpha for r in rows_a], dtype=torch.float32)
    alpha_b = torch.tensor([r.alpha for r in rows_b], dtype=torch.float32)

    trace_a = torch.tensor([r.trace_sigma for r in rows_a], dtype=torch.float32)
    trace_b = torch.tensor([r.trace_sigma for r in rows_b], dtype=torch.float32)

    alpha_test = permutation_test_mean_diff(
        alpha_a,
        alpha_b,
        num_perm=args.num_perm,
        seed=args.seed,
    )
    trace_test = permutation_test_mean_diff(
        trace_a,
        trace_b,
        num_perm=args.num_perm,
        seed=args.seed + 1,
    )

    print("\n[alpha mean difference test]")
    print(json.dumps(alpha_test, indent=2, ensure_ascii=False))
    print("\n[trace_sigma mean difference test]")
    print(json.dumps(trace_test, indent=2, ensure_ascii=False))

    output: dict[str, Any] = {
        "config": vars(args),
        "n_txt": n_txt,
        "lambda_txt": lambda_txt,
        "group_a_name": args.group_a_name,
        "group_b_name": args.group_b_name,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "rows_a": [asdict(r) for r in rows_a],
        "rows_b": [asdict(r) for r in rows_b],
        "alpha_test": alpha_test,
        "trace_sigma_test": trace_test,
    }

    # 可选：在真实图像 embedding 上比较 predictive logit variance
    if args.image_embeds_path is not None:
        image_embeds = load_image_embeddings(args.image_embeds_path).to(args.device)

        logits_var_a = compute_predictive_logit_variance(
            image_embeds=image_embeds,
            prompt_payload=payload_a,
            logit_scale=vlm.logit_scale,
            use_full_cov=args.use_full_cov,
        )  # [B, C_a]

        logits_var_b = compute_predictive_logit_variance(
            image_embeds=image_embeds,
            prompt_payload=payload_b,
            logit_scale=vlm.logit_scale,
            use_full_cov=args.use_full_cov,
        )  # [B, C_b]

        # 对每张图像，先在组内 prompt 上求平均 variance，再做配对检验
        per_image_mean_a = logits_var_a.mean(dim=1).detach().cpu()
        per_image_mean_b = logits_var_b.mean(dim=1).detach().cpu()

        paired_test = paired_sign_flip_test(
            per_image_mean_a,
            per_image_mean_b,
            num_perm=args.num_perm,
            seed=args.seed + 2,
        )

        pred_summary = {
            "group_a_predictive_var_mean": float(logits_var_a.mean().item()),
            "group_b_predictive_var_mean": float(logits_var_b.mean().item()),
            "group_a_per_image_mean_var_mean": float(per_image_mean_a.mean().item()),
            "group_b_per_image_mean_var_mean": float(per_image_mean_b.mean().item()),
            "paired_test_on_per_image_mean_var": paired_test,
            "num_images": int(image_embeds.shape[0]),
            "use_full_cov": bool(args.use_full_cov),
        }

        print("\n[predictive logit variance summary]")
        print(json.dumps(pred_summary, indent=2, ensure_ascii=False))

        output["predictive_logit_variance"] = pred_summary

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()