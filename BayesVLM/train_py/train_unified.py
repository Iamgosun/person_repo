from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_py.train_experiment import run_recipe_from_args


def normalize_recipe_name(recipe_name: str) -> str:
    key = str(recipe_name).strip().lower()
    if key in {"deterministic_coop_standard", "deterministic_coop"}:
        return "deterministic_coop"
    if key in {"text_only_bayes_coop", "vlm_adapter"}:
        return key
    raise ValueError(
        f"未知 recipe_name: {recipe_name}，可选值为 ['text_only_bayes_coop', 'vlm_adapter', 'deterministic_coop']"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe_name",
        type=str,
        required=True,
        help="训练配方名，可选：text_only_bayes_coop / vlm_adapter / deterministic_coop",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default=None,
        help="输出目录中的方法名标签；不传时默认等于 recipe_name。",
    )

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--local_model_path", type=str, default="./models/clip-vit-b32")
    parser.add_argument("--data_root", type=str, default="./datasets")

    parser.add_argument("--n_ctx", type=int, default=16)
    parser.add_argument("--ctx_init", type=str, default="a photo of a")
    parser.add_argument("--csc", action="store_true", default=False)
    parser.add_argument("--class_token_position", type=str, default="end")
    parser.add_argument("--shots_per_class", type=int, default=16)

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adam", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", default=False)

    parser.add_argument("--lr_scheduler", type=str, default=None, choices=["none", "cosine"])
    parser.add_argument("--warmup_epoch", type=int, default=0)
    parser.add_argument("--warmup_cons_lr", type=float, default=1e-5)

    parser.add_argument("--model_selection", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--selection_metric", type=str, default=None, choices=["loss", "acc", "nlpd", "ece"])
    parser.add_argument("--selection_mode", type=str, default="auto", choices=["auto", "min", "max"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--prediction_topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--disable_cache_image_features", action="store_true", default=False)
    parser.add_argument("--image_feature_cache_root", type=str, default="./cache/image_features")
    parser.add_argument("--rebuild_image_feature_cache", action="store_true", default=False)

    parser.add_argument("--use_data_augmentation", action="store_true", default=False)
    parser.add_argument("--use_augmented_train_cache", action="store_true", default=False)
    parser.add_argument("--train_aug_repeats", type=int, default=20)

    # text_only_bayes_coop 相关
    parser.add_argument("--hessian_dir", type=str, default=None)
    parser.add_argument("--pseudo_data_count", type=int, default=4)
    parser.add_argument("--lambda_txt_init", type=float, default=300.0)
    parser.add_argument("--lambda_opt_steps", type=int, default=500)
    parser.add_argument("--use_full_cov", action="store_true", default=False)
    parser.add_argument("--train_objective", type=str, default="hybrid", choices=["map", "bayes", "hybrid"])
    parser.add_argument("--hybrid_warmup_epochs", type=int, default=5)
    parser.add_argument("--map_loss_weight", type=float, default=1.0)
    parser.add_argument("--bayes_loss_weight", type=float, default=1.0)
    parser.add_argument("--ctx_reg_weight", type=float, default=1e-4)

    # vlm_adapter 相关
    parser.add_argument("--adapter_name", type=str, default="LP")
    parser.add_argument("--initialization", type=str, default="MEAN")
    parser.add_argument("--taskres_alpha", type=float, default=0.5)
    parser.add_argument("--clipa_ratio", type=float, default=0.2)
    parser.add_argument("--clipa_hidden_dim", type=int, default=0)
    parser.add_argument("--tipa_alpha", type=float, default=1.0)
    parser.add_argument("--tipa_beta", type=float, default=1.0)
    parser.add_argument("--gaussian_prior_sigma", type=float, default=0.01)
    parser.add_argument("--gaussian_mc_samples", type=int, default=3)
    parser.add_argument("--gaussian_anneal_start_epoch", type=int, default=20)

    # BayesAdapter 相关
    parser.add_argument("--bayesadapter_prior_sigma", type=float, default=0.01)
    parser.add_argument("--bayesadapter_train_mc_samples", type=int, default=3)
    parser.add_argument("--bayesadapter_eval_mc_samples", type=int, default=10)
    parser.add_argument("--bayesadapter_kl_scale_divisor", type=float, default=1000.0)

    # 选择 covariance 结构：
    # paper_scalar -> 与论文/当前实现一致，每类一个标量 sigma，协方差为 sigma_c^2 I_D
    # diag         -> 扩展版，每类一个 D 维对角 sigma
    parser.add_argument(
        "--bayesadapter_covariance_mode",
        type=str,
        default="paper_scalar",
        choices=["paper_scalar", "diag"],
    )

    # 选择先验来源：
    # base_text             -> 当前默认逻辑，使用 zero-shot text prototypes
    # text_only_bayes_coop  -> 用 text_only_bayes_coop 的 prompt posterior 重建先验
    parser.add_argument(
        "--bayesadapter_prior_source",
        type=str,
        default="base_text",
        choices=["base_text", "text_only_bayes_coop"],
    )
    parser.add_argument("--bayesadapter_text_only_run_dir", type=str, default="")
    parser.add_argument("--bayesadapter_text_only_ckpt", type=str, default="auto")

    return parser


def _validate_bayesadapter_args(args) -> None:
    if args.recipe_name != "vlm_adapter":
        return

    adapter_key = str(getattr(args, "adapter_name", "")).upper()
    if adapter_key not in {"BAYESADAPTER", "BAYES_ADAPTER"}:
        return

    prior_source = str(getattr(args, "bayesadapter_prior_source", "base_text")).lower()
    if prior_source == "text_only_bayes_coop":
        run_dir = str(getattr(args, "bayesadapter_text_only_run_dir", "")).strip()
        if not run_dir:
            raise ValueError(
                "当 adapter_name=BAYESADAPTER 且 bayesadapter_prior_source=text_only_bayes_coop 时，"
                "必须传 --bayesadapter_text_only_run_dir。"
            )


def prepare_args(args) -> None:
    args.recipe_name = normalize_recipe_name(args.recipe_name)
    if not args.method_name:
        args.method_name = args.recipe_name
    args.cache_image_features = not args.disable_cache_image_features
    args.train_aug_repeats = int(max(args.train_aug_repeats, 1))

    if args.recipe_name == "text_only_bayes_coop" and not args.hessian_dir:
        raise ValueError("text_only_bayes_coop 必须传 --hessian_dir。")

    _validate_bayesadapter_args(args)


def parse_and_run_fixed_recipe(recipe_name: str, parser: argparse.ArgumentParser | None = None) -> None:
    parser = parser or build_parser()
    args = parser.parse_args()
    args.recipe_name = normalize_recipe_name(recipe_name)
    if not getattr(args, "method_name", None):
        args.method_name = args.recipe_name
    args.cache_image_features = not args.disable_cache_image_features
    args.train_aug_repeats = int(max(args.train_aug_repeats, 1))

    if args.recipe_name == "text_only_bayes_coop" and not args.hessian_dir:
        raise ValueError("text_only_bayes_coop 必须传 --hessian_dir。")

    _validate_bayesadapter_args(args)
    run_recipe_from_args(args)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    prepare_args(args)
    run_recipe_from_args(args)