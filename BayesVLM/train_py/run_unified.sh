#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ============================================================
# 统一 shell 入口说明
# ------------------------------------------------------------
# 1) 真正只维护这一份脚本逻辑：run_unified.sh
# 2) 三个旧名字脚本只做薄封装，转发到这里
# 3) 通过 RECIPE_NAME 控制任务：
#       - text_only_bayes_coop
#       - deterministic_coop
#       - vlm_adapter
# 4) 支持命令行额外透传参数，例如：
#       bash train_py/run_unified.sh --epochs 50 --seed 2
# ============================================================

# =========================
# 任务选择
# =========================
RECIPE_NAME="${RECIPE_NAME:-text_only_bayes_coop}"

# =========================
# 基础路径配置
# =========================
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
MODEL_PATH="${MODEL_PATH:-./models/clip-vit-b32}"
HESSIAN_DIR="${HESSIAN_DIR:-./hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K}"
CACHE_ROOT="${CACHE_ROOT:-./cache/image_features}"

# =========================
# sweep 配置
# 支持这样传：
#   DATASETS="food101 cifar10"
#   SHOTS_PER_CLASS_LIST="1 2 4 8 16"
#   SEEDS="1 2 3"
#   METHODS="LP:RANDOM TR:TR TIPA:TIPA"
# =========================
IFS=' ' read -r -a DATASETS <<< "${DATASETS:-food101}"
IFS=' ' read -r -a SHOTS_PER_CLASS_LIST <<< "${SHOTS_PER_CLASS_LIST:-16}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-1}"
IFS=' ' read -r -a METHODS <<< "${METHODS:-LP:RANDOM}"

# =========================
# 全局训练配置
# =========================
MODEL_STR="${MODEL_STR:-clip-base}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_py/train_unified.py}"

NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PREDICTION_TOPK="${PREDICTION_TOPK:-5}"

# 缓存相关
REBUILD_IMAGE_CACHE="${REBUILD_IMAGE_CACHE:-0}"
DISABLE_IMAGE_CACHE="${DISABLE_IMAGE_CACHE:-0}"

# 通用 prompt 参数
N_CTX="${N_CTX:-16}"
CTX_INIT="${CTX_INIT:-}"
CSC="${CSC:-0}"
CLASS_TOKEN_POSITION="${CLASS_TOKEN_POSITION:-end}"

# 通用优化器 / 调度器参数
LR="${LR:-0.002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
EPOCHS="${EPOCHS:-200}"
OPTIMIZER="${OPTIMIZER:-}"
MOMENTUM="${MOMENTUM:-0.9}"
NESTEROV="${NESTEROV:-0}"
LR_SCHEDULER="${LR_SCHEDULER:-}"
WARMUP_EPOCH="${WARMUP_EPOCH:-0}"
WARMUP_CONS_LR="${WARMUP_CONS_LR:-1e-5}"

# checkpoint 选择
MODEL_SELECTION="${MODEL_SELECTION:-}"
SELECTION_METRIC="${SELECTION_METRIC:-}"
SELECTION_MODE="${SELECTION_MODE:-}"

# text_only_bayes_coop 相关
PSEUDO_DATA_COUNT="${PSEUDO_DATA_COUNT:-4}"
LAMBDA_TXT_INIT="${LAMBDA_TXT_INIT:-300.0}"
LAMBDA_OPT_STEPS="${LAMBDA_OPT_STEPS:-1000}"
USE_FULL_COV="${USE_FULL_COV:-0}"
TRAIN_OBJECTIVE="${TRAIN_OBJECTIVE:-bayes}"
HYBRID_WARMUP_EPOCHS="${HYBRID_WARMUP_EPOCHS:-5}"
MAP_LOSS_WEIGHT="${MAP_LOSS_WEIGHT:-1.0}"
BAYES_LOSS_WEIGHT="${BAYES_LOSS_WEIGHT:-1.0}"
CTX_REG_WEIGHT="${CTX_REG_WEIGHT:-1e-4}"

# vlm_adapter 相关
# METHODS 形如：
#   METHODS="LP:RANDOM TR:TR TIPA:TIPA"
TASKRES_ALPHA="${TASKRES_ALPHA:-0.5}"
CLIPA_RATIO="${CLIPA_RATIO:-0.2}"
CLIPA_HIDDEN_DIM="${CLIPA_HIDDEN_DIM:-0}"
TIPA_ALPHA="${TIPA_ALPHA:-1.0}"
TIPA_BETA="${TIPA_BETA:-1.0}"
GAUSSIAN_PRIOR_SIGMA="${GAUSSIAN_PRIOR_SIGMA:-0.01}"
GAUSSIAN_MC_SAMPLES="${GAUSSIAN_MC_SAMPLES:-3}"
GAUSSIAN_ANNEAL_START_EPOCH="${GAUSSIAN_ANNEAL_START_EPOCH:-20}"

# 允许从命令行直接追加任意 train_unified.py 参数
EXTRA_ARGS=("$@")

append_common_optional_args() {
  if [[ "${REBUILD_IMAGE_CACHE}" -eq 1 ]]; then
    EXTRA_ARGS+=("--rebuild_image_feature_cache")
  fi

  if [[ "${DISABLE_IMAGE_CACHE}" -eq 1 ]]; then
    EXTRA_ARGS+=("--disable_cache_image_features")
  fi

  if [[ "${CSC}" -eq 1 ]]; then
    EXTRA_ARGS+=("--csc")
  fi

  if [[ "${USE_FULL_COV}" -eq 1 ]]; then
    EXTRA_ARGS+=("--use_full_cov")
  fi

  if [[ "${NESTEROV}" -eq 1 ]]; then
    EXTRA_ARGS+=("--nesterov")
  fi

  if [[ -n "${OPTIMIZER}" ]]; then
    EXTRA_ARGS+=("--optimizer" "${OPTIMIZER}")
  fi

  if [[ -n "${LR_SCHEDULER}" ]]; then
    EXTRA_ARGS+=("--lr_scheduler" "${LR_SCHEDULER}")
  fi

  if [[ -n "${MODEL_SELECTION}" ]]; then
    EXTRA_ARGS+=("--model_selection" "${MODEL_SELECTION}")
  fi

  if [[ -n "${SELECTION_METRIC}" ]]; then
    EXTRA_ARGS+=("--selection_metric" "${SELECTION_METRIC}")
  fi

  if [[ -n "${SELECTION_MODE}" ]]; then
    EXTRA_ARGS+=("--selection_mode" "${SELECTION_MODE}")
  fi
}

print_common_header() {
  echo "============================================================"
  echo "recipe_name=${RECIPE_NAME}"
  echo "dataset=$1"
  echo "shots=$2"
  echo "seed=$3"
  echo "model=${MODEL_STR}"
  echo "model_path=${MODEL_PATH}"
  echo "data_root=${DATA_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "cache_root=${CACHE_ROOT}"
  echo "rebuild_image_cache=${REBUILD_IMAGE_CACHE}"
  echo "disable_image_cache=${DISABLE_IMAGE_CACHE}"
  echo "============================================================"
}

run_text_only_bayes_coop() {
  local dataset="$1"
  local shots="$2"
  local seed="$3"

  local method_name="${METHOD_NAME:-text_only_bayes_coop_${TRAIN_OBJECTIVE}}"

  print_common_header "${dataset}" "${shots}" "${seed}"
  echo "method_name=${method_name}"
  echo "train_objective=${TRAIN_OBJECTIVE}"
  echo "hybrid_warmup_epochs=${HYBRID_WARMUP_EPOCHS}"
  echo "map_loss_weight=${MAP_LOSS_WEIGHT}"
  echo "bayes_loss_weight=${BAYES_LOSS_WEIGHT}"
  echo "ctx_reg_weight=${CTX_REG_WEIGHT}"
  echo "use_full_cov=${USE_FULL_COV}"
  echo "warmup_epoch=${WARMUP_EPOCH}"
  echo "warmup_cons_lr=${WARMUP_CONS_LR}"
  echo "model_selection=${MODEL_SELECTION:-last}"
  echo

  local cmd=(
    "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}"
    --recipe_name text_only_bayes_coop
    --method_name "${method_name}"
    --dataset "${dataset}"
    --model "${MODEL_STR}"
    --local_model_path "${MODEL_PATH}"
    --data_root "${DATA_ROOT}"
    --hessian_dir "${HESSIAN_DIR}"
    --image_feature_cache_root "${CACHE_ROOT}"
    --pseudo_data_count "${PSEUDO_DATA_COUNT}"
    --lambda_txt_init "${LAMBDA_TXT_INIT}"
    --lambda_opt_steps "${LAMBDA_OPT_STEPS}"
    --n_ctx "${N_CTX}"
    --ctx_init "${CTX_INIT}"
    --class_token_position "${CLASS_TOKEN_POSITION}"
    --shots_per_class "${shots}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --momentum "${MOMENTUM}"
    --warmup_epoch "${WARMUP_EPOCH}"
    --warmup_cons_lr "${WARMUP_CONS_LR}"
    --train_objective "${TRAIN_OBJECTIVE}"
    --hybrid_warmup_epochs "${HYBRID_WARMUP_EPOCHS}"
    --map_loss_weight "${MAP_LOSS_WEIGHT}"
    --bayes_loss_weight "${BAYES_LOSS_WEIGHT}"
    --ctx_reg_weight "${CTX_REG_WEIGHT}"
    --save_dir "${OUTPUT_ROOT}"
    --prediction_topk "${PREDICTION_TOPK}"
    --seed "${seed}"
    --device "${DEVICE}"
  )

  # 与旧版 shell 保持一致：text_only 默认 last + SGD + cosine
  if [[ -z "${MODEL_SELECTION}" ]]; then
    cmd+=(--model_selection last)
  fi
  if [[ -z "${OPTIMIZER}" ]]; then
    cmd+=(--optimizer sgd)
  fi
  if [[ -z "${LR_SCHEDULER}" ]]; then
    cmd+=(--lr_scheduler cosine)
  fi
  if [[ -z "${SELECTION_METRIC}" ]]; then
    cmd+=(--selection_metric acc)
  fi

  cmd+=("${EXTRA_ARGS[@]}")
  "${cmd[@]}"
}

run_deterministic_coop() {
  local dataset="$1"
  local shots="$2"
  local seed="$3"

  local method_name="${METHOD_NAME:-deterministic_coop_standard}"

  print_common_header "${dataset}" "${shots}" "${seed}"
  echo "method_name=${method_name}"
  echo "warmup_epoch=${WARMUP_EPOCH}"
  echo "warmup_cons_lr=${WARMUP_CONS_LR}"
  echo "model_selection=${MODEL_SELECTION:-best}"
  echo

  local cmd=(
    "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}"
    --recipe_name deterministic_coop
    --method_name "${method_name}"
    --dataset "${dataset}"
    --model "${MODEL_STR}"
    --local_model_path "${MODEL_PATH}"
    --data_root "${DATA_ROOT}"
    --image_feature_cache_root "${CACHE_ROOT}"
    --n_ctx "${N_CTX}"
    --ctx_init "${CTX_INIT}"
    --class_token_position "${CLASS_TOKEN_POSITION}"
    --shots_per_class "${shots}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --momentum "${MOMENTUM}"
    --warmup_epoch "${WARMUP_EPOCH}"
    --warmup_cons_lr "${WARMUP_CONS_LR}"
    --save_dir "${OUTPUT_ROOT}"
    --prediction_topk "${PREDICTION_TOPK}"
    --seed "${seed}"
    --device "${DEVICE}"
  )

  # 与旧版 deterministic CoOp 一致：best + SGD + cosine
  if [[ -z "${MODEL_SELECTION}" ]]; then
    cmd+=(--model_selection best)
  fi
  if [[ -z "${OPTIMIZER}" ]]; then
    cmd+=(--optimizer sgd)
  fi
  if [[ -z "${LR_SCHEDULER}" ]]; then
    cmd+=(--lr_scheduler cosine)
  fi
  if [[ -z "${SELECTION_METRIC}" ]]; then
    cmd+=(--selection_metric acc)
  fi

  cmd+=("${EXTRA_ARGS[@]}")
  "${cmd[@]}"
}

run_vlm_adapter() {
  local dataset="$1"
  local adapter_name="$2"
  local initialization="$3"
  local shots="$4"
  local seed="$5"

  local method_name="${METHOD_NAME:-vlm_adapter}"

  print_common_header "${dataset}" "${shots}" "${seed}"
  echo "method_name=${method_name}"
  echo "adapter_name=${adapter_name}"
  echo "initialization=${initialization}"
  echo "model_selection=${MODEL_SELECTION:-best}"
  echo

  local cmd=(
    "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}"
    --recipe_name vlm_adapter
    --method_name "${method_name}"
    --dataset "${dataset}"
    --model "${MODEL_STR}"
    --local_model_path "${MODEL_PATH}"
    --data_root "${DATA_ROOT}"
    --image_feature_cache_root "${CACHE_ROOT}"
    --adapter_name "${adapter_name}"
    --initialization "${initialization}"
    --shots_per_class "${shots}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --save_dir "${OUTPUT_ROOT}"
    --prediction_topk "${PREDICTION_TOPK}"
    --seed "${seed}"
    --device "${DEVICE}"
    --taskres_alpha "${TASKRES_ALPHA}"
    --clipa_ratio "${CLIPA_RATIO}"
    --clipa_hidden_dim "${CLIPA_HIDDEN_DIM}"
    --tipa_alpha "${TIPA_ALPHA}"
    --tipa_beta "${TIPA_BETA}"
    --gaussian_prior_sigma "${GAUSSIAN_PRIOR_SIGMA}"
    --gaussian_mc_samples "${GAUSSIAN_MC_SAMPLES}"
    --gaussian_anneal_start_epoch "${GAUSSIAN_ANNEAL_START_EPOCH}"
    --hessian_dir "${HESSIAN_DIR}"
    --pseudo_data_count "${PSEUDO_DATA_COUNT}"
  )

  # 与当前 adapter recipe 一致：best + AdamW + no scheduler + val loss
  if [[ -z "${MODEL_SELECTION}" ]]; then
    cmd+=(--model_selection best)
  fi
  if [[ -z "${OPTIMIZER}" ]]; then
    cmd+=(--optimizer adamw)
  fi
  if [[ -z "${LR_SCHEDULER}" ]]; then
    cmd+=(--lr_scheduler none)
  fi
  if [[ -z "${SELECTION_METRIC}" ]]; then
    cmd+=(--selection_metric loss)
  fi

  cmd+=("${EXTRA_ARGS[@]}")
  "${cmd[@]}"
}

append_common_optional_args

case "${RECIPE_NAME}" in
  text_only_bayes_coop)
    for dataset in "${DATASETS[@]}"; do
      for shots in "${SHOTS_PER_CLASS_LIST[@]}"; do
        for seed in "${SEEDS[@]}"; do
          run_text_only_bayes_coop "${dataset}" "${shots}" "${seed}"
        done
      done
    done
    ;;
  deterministic_coop|deterministic_coop_standard)
    for dataset in "${DATASETS[@]}"; do
      for shots in "${SHOTS_PER_CLASS_LIST[@]}"; do
        for seed in "${SEEDS[@]}"; do
          run_deterministic_coop "${dataset}" "${shots}" "${seed}"
        done
      done
    done
    ;;
  vlm_adapter)
    for dataset in "${DATASETS[@]}"; do
      for method in "${METHODS[@]}"; do
        IFS=':' read -r adapter_name initialization <<< "${method}"
        for shots in "${SHOTS_PER_CLASS_LIST[@]}"; do
          for seed in "${SEEDS[@]}"; do
            run_vlm_adapter "${dataset}" "${adapter_name}" "${initialization}" "${shots}" "${seed}"
          done
        done
      done
    done
    ;;
  *)
    echo "未知 RECIPE_NAME=${RECIPE_NAME}"
    echo "可选值：text_only_bayes_coop / deterministic_coop / vlm_adapter"
    exit 1
    ;;
esac

echo "All runs finished."