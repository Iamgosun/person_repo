from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_py.train_experiment import run_recipe_from_args
from train_py.xml_experiment import build_resolved_run_namespaces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 XML plan 读取实验配置并运行。"
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="XML plan 文件路径，例如 configs/plans/vlm_adapter_bayesadapter_diag_textonly.xml",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印展开后的最终配置，不真正运行训练。",
    )
    parser.add_argument(
        "--only_index",
        type=int,
        default=None,
        help="只运行某一条 experiment（从 1 开始计数）。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    namespaces = build_resolved_run_namespaces(args.plan)

    if args.only_index is not None:
        if args.only_index < 1 or args.only_index > len(namespaces):
            raise ValueError(
                f"--only_index 超出范围：当前共有 {len(namespaces)} 条 experiment，"
                f"但你传的是 {args.only_index}"
            )
        namespaces = [namespaces[args.only_index - 1]]

    if args.dry_run:
        for idx, ns in enumerate(namespaces, start=1):
            print("=" * 80)
            print(f"[dry-run] experiment {idx}/{len(namespaces)}")
            print(json.dumps(vars(ns), ensure_ascii=False, indent=2))
        return

    total = len(namespaces)
    for idx, ns in enumerate(namespaces, start=1):
        print("=" * 80)
        print(f"[xml-runner] start experiment {idx}/{total}")
        print(
            json.dumps(
                {
                    "recipe_name": ns.recipe_name,
                    "method_name": ns.method_name,
                    "dataset": ns.dataset,
                    "shots_per_class": ns.shots_per_class,
                    "seed": ns.seed,
                    "adapter_name": getattr(ns, "adapter_name", None),
                    "initialization": getattr(ns, "initialization", None),
                    "bayesadapter_text_only_run_dir": getattr(ns, "bayesadapter_text_only_run_dir", None),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        run_recipe_from_args(ns)


if __name__ == "__main__":
    main()