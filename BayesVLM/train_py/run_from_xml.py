from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_py.runtime.eval_runtime import run_family_eval
from train_py.runtime.train_runtime import run_family_train
from train_py.xml_experiment import build_resolved_run_namespaces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment plans from XML.")
    parser.add_argument("--plan", type=str, required=True, help="XML experiment plan path")
    parser.add_argument("--stage", type=str, default="all", choices=["train", "eval", "all"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--only_index", type=int, default=None)
    return parser


def _brief(ns, idx: int, total: int, stage: str) -> None:
    print("=" * 80)
    print(f"[xml-runner] stage={stage} experiment {idx}/{total}")
    payload = {
        "family": ns.family,
        "variant": ns.variant,
        "protocol": ns.protocol,
        "evaluation_tasks": ns.evaluation_tasks,
        "dataset": ns.dataset,
        "shots_per_class": ns.shots_per_class,
        "seed": ns.seed,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    namespaces = build_resolved_run_namespaces(args.plan)
    if args.only_index is not None:
        if args.only_index < 1 or args.only_index > len(namespaces):
            raise ValueError(
                f"--only_index out of range: got {args.only_index}, total {len(namespaces)}"
            )
        namespaces = [namespaces[args.only_index - 1]]

    if args.dry_run:
        for idx, ns in enumerate(namespaces, start=1):
            print("=" * 80)
            print(f"[dry-run] stage={args.stage} experiment {idx}/{len(namespaces)}")
            print(json.dumps(vars(ns), ensure_ascii=False, indent=2))
        return

    total = len(namespaces)
    for idx, ns in enumerate(namespaces, start=1):
        if args.stage in {"train", "all"}:
            _brief(ns, idx, total, stage="train")
            run_family_train(ns)
        if args.stage in {"eval", "all"}:
            _brief(ns, idx, total, stage="eval")
            run_family_eval(ns)


if __name__ == "__main__":
    main()
