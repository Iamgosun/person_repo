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
        recipe_name="text_only_bayes_coop",
        method_name="text_only_bayes_coop",
    )
    parse_and_run_fixed_recipe("text_only_bayes_coop", parser=parser)
