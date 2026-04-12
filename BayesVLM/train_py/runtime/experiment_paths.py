from __future__ import annotations

from pathlib import Path


def build_run_dir(args) -> Path:
    root = Path(args.save_dir)
    output_name = str(getattr(args, "output_name", args.family)).strip()
    root = root / output_name / args.protocol / args.dataset
    if str(args.variant).strip().lower() != "default":
        root = root / str(args.variant)
    family_parts = getattr(args, "_family_run_path_parts", None)
    if family_parts:
        for part in family_parts:
            root = root / str(part)
    root = root / f"seed_{args.seed}"
    return root.resolve()
