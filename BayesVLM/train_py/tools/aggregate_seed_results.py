from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _safe_mean(xs: list[float]) -> float | None:
    return float(mean(xs)) if xs else None


def _safe_std(xs: list[float]) -> float:
    return float(stdev(xs)) if len(xs) > 1 else 0.0


def _flatten_numeric_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    """
    递归提取 dict 里的所有 numeric scalar。
    例:
      {"acc": 0.8, "cov_99": 0.1}
    -> {"acc": 0.8, "cov_99": 0.1}

      {"test": {"metrics": {"acc": 0.8}}}
    -> {"test.metrics.acc": 0.8}
    """
    out: dict[str, float] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric_metrics(v, prefix=key))
        return out

    if _is_number(obj):
        if math.isfinite(float(obj)):
            out[prefix] = float(obj)
        return out

    return out


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_run_meta_from_config(run_dir: Path) -> dict[str, Any]:
    """
    summary.json 当前不一定包含 shots_per_class / output_name，
    所以优先从 run_dir/config/config.json 读。
    """
    config_path = run_dir / "config" / "config.json"
    if not config_path.exists():
        return {
            "shots_per_class": None,
            "output_name": None,
            "family": None,
            "variant": None,
            "protocol": None,
            "dataset": None,
            "seed": None,
            "model_selection": None,
        }

    cfg = _load_json(config_path)
    return {
        "shots_per_class": cfg.get("shots_per_class"),
        "output_name": cfg.get("output_name", cfg.get("family")),
        "family": cfg.get("family"),
        "variant": cfg.get("variant"),
        "protocol": cfg.get("protocol"),
        "dataset": cfg.get("dataset"),
        "seed": cfg.get("seed"),
        "model_selection": cfg.get("model_selection"),
    }


def _extract_seed_level_row(summary_path: Path) -> dict[str, Any]:
    """
    从单个 run_dir/summary.json 提取一行 seed-level 记录。
    自动收集 summary["classification"][split]["metrics"] 下所有 numeric metrics。
    """
    summary = _load_json(summary_path)
    run_dir = Path(summary["run_dir"])
    cfg_meta = _infer_run_meta_from_config(run_dir)

    classification = summary.get("classification", {})
    metric_payload: dict[str, float] = {}

    for split_name, split_result in classification.items():
        if not isinstance(split_result, dict):
            continue
        split_metrics = split_result.get("metrics", {})
        flat = _flatten_numeric_metrics(split_metrics)
        for k, v in flat.items():
            metric_payload[f"{split_name}.{k}"] = v

    row: dict[str, Any] = {
        "summary_path": str(summary_path.resolve()),
        "run_dir": str(run_dir),
        "output_name": cfg_meta["output_name"],
        "family": summary.get("family", cfg_meta["family"]),
        "variant": summary.get("variant", cfg_meta["variant"]),
        "protocol": summary.get("protocol", cfg_meta["protocol"]),
        "dataset": summary.get("dataset", cfg_meta["dataset"]),
        "shots_per_class": cfg_meta["shots_per_class"],
        "seed": summary.get("seed", cfg_meta["seed"]),
        "model_selection": summary.get("model_selection", cfg_meta["model_selection"]),
        "selected_checkpoint": summary.get("selected_checkpoint"),
        "selected_epoch": summary.get("selected_epoch"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }
    row.update(metric_payload)
    return row


def _group_key(row: dict[str, Any]) -> tuple:
    """
    除 seed 外的分组键。
    你后面如果需要把 initialization / backbone / adapter 也纳入分组，
    可以继续往这里加。
    """
    return (
        row.get("output_name"),
        row.get("family"),
        row.get("variant"),
        row.get("protocol"),
        row.get("dataset"),
        row.get("shots_per_class"),
        row.get("model_selection"),
        row.get("selected_checkpoint"),
    )


def _aggregate_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[_group_key(row)].append(row)

    agg_rows: list[dict[str, Any]] = []

    for key, bucket in grouped.items():
        (
            output_name,
            family,
            variant,
            protocol,
            dataset,
            shots_per_class,
            model_selection,
            selected_checkpoint,
        ) = key

        seeds = sorted(
            [int(x["seed"]) for x in bucket if x.get("seed") is not None]
        )

        base: dict[str, Any] = {
            "output_name": output_name,
            "family": family,
            "variant": variant,
            "protocol": protocol,
            "dataset": dataset,
            "shots_per_class": shots_per_class,
            "model_selection": model_selection,
            "selected_checkpoint": selected_checkpoint,
            "n_seeds": len(bucket),
            "seeds": ",".join(str(x) for x in seeds),
        }

        # 聚合所有 numeric metric keys
        metric_keys = set()
        for row in bucket:
            for k, v in row.items():
                if k in base:
                    continue
                if _is_number(v):
                    metric_keys.add(k)

        for mk in sorted(metric_keys):
            vals = [float(r[mk]) for r in bucket if mk in r and _is_number(r[mk])]
            if len(vals) == 0:
                continue
            base[f"{mk}__mean"] = _safe_mean(vals)
            base[f"{mk}__std"] = _safe_std(vals)
            base[f"{mk}__min"] = min(vals)
            base[f"{mk}__max"] = max(vals)

        agg_rows.append(base)

    agg_rows.sort(
        key=lambda x: (
            str(x.get("output_name")),
            str(x.get("family")),
            str(x.get("variant")),
            str(x.get("protocol")),
            str(x.get("dataset")),
            -1 if x.get("shots_per_class") is None else int(x["shots_per_class"]),
        )
    )
    return agg_rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results across seeds.")
    parser.add_argument("--root", type=str, default="./output", help="Search root for summary.json")
    parser.add_argument(
        "--summary_glob",
        type=str,
        default="**/summary.json",
        help="Glob pattern relative to --root",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./output/_aggregated",
        help="Directory to save aggregated results",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()

    summary_paths = sorted(root.glob(args.summary_glob))
    if len(summary_paths) == 0:
        raise FileNotFoundError(
            f"No summary.json found under root={root} with pattern={args.summary_glob}"
        )

    seed_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for sp in summary_paths:
        try:
            row = _extract_seed_level_row(sp)
            seed_rows.append(row)
        except Exception as e:
            skipped.append({"summary_path": str(sp), "error": repr(e)})

    agg_rows = _aggregate_rows(seed_rows)

    _write_json(out_dir / "seed_level_rows.json", seed_rows)
    _write_csv(out_dir / "seed_level_rows.csv", seed_rows)

    _write_json(out_dir / "grouped_seed_summary.json", agg_rows)
    _write_csv(out_dir / "grouped_seed_summary.csv", agg_rows)

    if skipped:
        _write_json(out_dir / "skipped.json", skipped)

    print("=" * 80)
    print(f"[aggregate] root={root}")
    print(f"[aggregate] found_summary_files={len(summary_paths)}")
    print(f"[aggregate] parsed_seed_rows={len(seed_rows)}")
    print(f"[aggregate] grouped_rows={len(agg_rows)}")
    print(f"[aggregate] out_dir={out_dir}")
    if skipped:
        print(f"[aggregate] skipped={len(skipped)} (see skipped.json)")
    print("=" * 80)


if __name__ == "__main__":
    main()