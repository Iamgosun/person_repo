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
#   DRY_RUN
#   ONLY_INDEX
#
# 说明：
#   1) PLAN_NAME 对应 configs/plans/ 下的某个 XML 文件名（不带 .xml）
#   2) DRY_RUN=1 时，只打印展开后的最终配置，不真正训练
#   3) ONLY_INDEX 非空时，只跑第 N 条 experiment（从 1 开始）
#
# 例如：
#   PLAN_NAME="text_only_bayes_coop_bayes"
#   PLAN_NAME="vlm_adapter_bayesadapter_diag_textonly"
# ============================================================

# ----------------------------
# 1) 选哪个实验 plan
# ----------------------------
PLAN_NAME="vlm_adapter_bayesadapter_diag_textonly"
# PLAN_NAME="deterministic_coop"
# PLAN_NAME="text_only_bayes_coop_bayes"
# PLAN_NAME="vlm_adapter_bayesadapter_diag_textonly"
# PLAN_NAME="vlm_adapter_lp_mean"
# PLAN_NAME="vlm_adapter_tr"
# PLAN_NAME="vlm_adapter_tipa"
# PLAN_NAME="vlm_adapter_crossmodal"
# PLAN_NAME="vlm_adapter_gaussian_per_class"




# ----------------------------
# 2) 是否只做 dry-run
# ----------------------------
DRY_RUN=0
# DRY_RUN=1

# ----------------------------
# 3) 是否只跑某一条 experiment
#    留空 -> 跑 plan 里的全部 experiment
#    例如 ONLY_INDEX="2" -> 只跑第 2 条
# ----------------------------
ONLY_INDEX=""
# ONLY_INDEX="1"

# ----------------------------
# 4) Python 与 runner
# ----------------------------
PYTHON_BIN="python"
RUNNER_SCRIPT="train_py/run_from_xml.py"
PLAN_DIR="configs/plans"

# ============================================================
# 下面一般不用改
# ============================================================

# 支持两种写法：
#   PLAN_NAME="xxx"         -> 自动拼成 configs/plans/xxx.xml
#   PLAN_NAME="a/b/c.xml"   -> 直接当成相对/绝对路径
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

CMD=(
  "${PYTHON_BIN}" -u "${RUNNER_SCRIPT}"
  --plan "${PLAN_PATH}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  CMD+=(--dry_run)
fi

if [[ -n "${ONLY_INDEX}" ]]; then
  CMD+=(--only_index "${ONLY_INDEX}")
fi

# 允许你在命令行后面额外加 runner 参数
# 例如：
#   bash train_py/run_unified.sh --dry_run
# 但如果你平时不需要，可以完全忽略
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "============================================================"
echo "[launcher] ROOT_DIR=${ROOT_DIR}"
echo "[launcher] PLAN_PATH=${PLAN_PATH}"
echo "[launcher] DRY_RUN=${DRY_RUN}"
echo "[launcher] ONLY_INDEX=${ONLY_INDEX:-<all>}"
echo "============================================================"

"${CMD[@]}"