#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ============================================================
# Unified XML launcher
# ------------------------------------------------------------
# Only modify these values in daily usage:
#   PLAN_NAME
#   STAGE
#   DRY_RUN
#   ONLY_INDEX
#
# PLAN_NAME points to configs/plans/<name>.xml (or an absolute path)
# STAGE: train / eval / all
# DRY_RUN=1 prints resolved configs without running
# ONLY_INDEX="N" runs only the N-th experiment (1-based)
# ============================================================
 # vmfproto_id sample_id
PLAN_NAME="vmfproto_id"
STAGE="all"
DRY_RUN=0
ONLY_INDEX=""

PYTHON_BIN="python"
RUNNER_SCRIPT="train_py/run_from_xml.py"
PLAN_DIR="configs/plans"

if [[ "${PLAN_NAME}" == *.xml ]]; then
  if [[ "${PLAN_NAME}" = /* ]]; then
    PLAN_PATH="${PLAN_NAME}"
  else
    PLAN_PATH="${ROOT_DIR}/${PLAN_NAME}"
  fi
else
  PLAN_PATH="${ROOT_DIR}/${PLAN_DIR}/${PLAN_NAME}.xml"
fi

if [[ ! -f "${PLAN_PATH}" ]]; then
  echo "[ERROR] plan file not found: ${PLAN_PATH}"
  exit 1
fi

case "${STAGE}" in
  train|eval|all)
    ;;
  *)
    echo "[ERROR] STAGE must be train / eval / all, got: ${STAGE}"
    exit 1
    ;;
esac

CMD=(
  "${PYTHON_BIN}" -u "${RUNNER_SCRIPT}"
  --plan "${PLAN_PATH}"
  --stage "${STAGE}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  CMD+=(--dry_run)
fi

if [[ -n "${ONLY_INDEX}" ]]; then
  CMD+=(--only_index "${ONLY_INDEX}")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "============================================================"
echo "[launcher] ROOT_DIR=${ROOT_DIR}"
echo "[launcher] PLAN_PATH=${PLAN_PATH}"
echo "[launcher] STAGE=${STAGE}"
echo "[launcher] DRY_RUN=${DRY_RUN}"
echo "[launcher] ONLY_INDEX=${ONLY_INDEX:-<all>}"
echo "============================================================"

"${CMD[@]}"
