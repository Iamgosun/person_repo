from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_py.train_unified import build_parser, parse_and_run_fixed_recipe


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(
        recipe_name="vlm_adapter",
        method_name="vlm_adapter",
    )
    parse_and_run_fixed_recipe("vlm_adapter", parser=parser)
