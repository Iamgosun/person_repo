#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ============================================================
# 统一 XML 启动脚本
# ------------------------------------------------------------
# 日常使用：
#   bash train_py/run_unified.sh
#
# 你平时只需要改下面这几个变量：
#   PLAN_NAME
#   STAGE
#   DRY_RUN
#   ONLY_INDEX
#
# 说明：
#   1) PLAN_NAME 对应 configs/plans/ 下的某个 XML 文件名（不带 .xml）
#   2) STAGE 可选 train / eval / all
#   3) DRY_RUN=1 时，只打印展开后的最终配置，不真正训练/评估
#   4) ONLY_INDEX 非空时，只跑第 N 条 experiment（从 1 开始）
# ============================================================

# ----------------------------
# 1) 选哪个实验 plan
# ----------------------------
PLAN_NAME="vlm_adapter_bayesadapter_diag_textonly"
# PLAN_NAME="deterministic_coop"
# PLAN_NAME="text_only_bayes_coop_bayes"
# vlm_adapter_bayesadapter_diag_textonly
# ----------------------------
# 2) 运行阶段
# ----------------------------
STAGE="all"
# STAGE="train"
# STAGE="eval"

# ----------------------------
# 3) 是否只做 dry-run
# ----------------------------
DRY_RUN=0
# DRY_RUN=1

# ----------------------------
# 4) 是否只跑某一条 experiment
# ----------------------------
ONLY_INDEX=""
# ONLY_INDEX="1"

# ----------------------------
# 5) Python 与 runner
# ----------------------------
PYTHON_BIN="python"
RUNNER_SCRIPT="train_py/run_from_xml.py"
PLAN_DIR="configs/plans"

# ============================================================
# 下面一般不用改
# ============================================================

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
  echo "[ERROR] 找不到 plan 文件: ${PLAN_PATH}"
  exit 1
fi

case "${STAGE}" in
  train|eval|all)
    ;;
  *)
    echo "[ERROR] STAGE 必须是 train / eval / all，当前值为: ${STAGE}"
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
