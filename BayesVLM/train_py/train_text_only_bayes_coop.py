from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_py.train_experiment import run_text_only_bayes_coop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--hessian_dir", type=str, required=True)
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
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--method_name", type=str, default="text_only_bayes_coop")
    parser.add_argument("--prediction_topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--disable_cache_image_features", action="store_true", default=False)
    parser.add_argument("--image_feature_cache_root", type=str, default="./cache/image_features")
    parser.add_argument("--rebuild_image_feature_cache", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.cache_image_features = not args.disable_cache_image_features
    run_text_only_bayes_coop(args)