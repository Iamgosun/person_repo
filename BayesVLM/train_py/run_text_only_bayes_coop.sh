#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_NAME="text_only_bayes_coop" bash "${SCRIPT_DIR}/run_unified.sh" "$@"
