#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# =========================
# 基础路径配置
# =========================
OUTPUT_ROOT="./output"
DATA_ROOT="./datasets"
MODEL_PATH="./models/clip-vit-b32"
HESSIAN_DIR="./hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"
CACHE_ROOT="./cache/image_features"

# =========================
# 运行配置
# =========================
DATASETS=("food101")
SHOTS_PER_CLASS_LIST=(1)
SEEDS=(1)

MODEL_STR="clip-base"
DEVICE="cuda"
PYTHON_BIN="python"
TRAIN_SCRIPT="train_py/train_text_only_bayes_coop.py"

# =========================
# Bayes / Hessian 相关
# =========================
PSEUDO_DATA_COUNT=4
LAMBDA_TXT_INIT=300.0
LAMBDA_OPT_STEPS=1000

# =========================
# CoOp prompt 相关
# =========================
N_CTX=16
CTX_INIT=""
CSC=0
CLASS_TOKEN_POSITION="end"

# =========================
# 训练目标相关
# train_objective 可选: map / bayes / hybrid
# hybrid 表示: 先 MAP warmup，再联合优化 MAP + Bayes + prompt regularization
# =========================
TRAIN_OBJECTIVE="bayes"
HYBRID_WARMUP_EPOCHS=5
MAP_LOSS_WEIGHT=1.0
BAYES_LOSS_WEIGHT=1.0
CTX_REG_WEIGHT=1e-4

# method_name 默认带上 objective，避免不同目标函数的结果写到同一路径里
METHOD_NAME="text_only_bayes_coop_${TRAIN_OBJECTIVE}"

# 模型选择方式: best / last
MODEL_SELECTION="last"

# =========================
# 学习率调度相关
# warmup_epoch = 0 时表示关闭 warmup，仅使用 cosine schedule
# =========================
WARMUP_EPOCH=1
WARMUP_CONS_LR=1e-5

# =========================
# 优化器 / 训练轮数
# =========================
LR=0.002
WEIGHT_DECAY=0
EPOCHS=200
BATCH_SIZE=256
NUM_WORKERS=8

# =========================
# 推理与缓存开关
# =========================
USE_FULL_COV=0
PREDICTION_TOPK=5
REBUILD_IMAGE_CACHE=0
DISABLE_IMAGE_CACHE=0

EXTRA_ARGS=()

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

for DATASET in "${DATASETS[@]}"; do
  for SHOTS_PER_CLASS in "${SHOTS_PER_CLASS_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "=========================================="
      echo "开始运行 Text-only Bayes CoOp"
      echo "method_name=${METHOD_NAME}"
      echo "dataset=${DATASET}"
      echo "shots=${SHOTS_PER_CLASS}"
      echo "seed=${SEED}"
      echo "train_objective=${TRAIN_OBJECTIVE}"
      echo "hybrid_warmup_epochs=${HYBRID_WARMUP_EPOCHS}"
      echo "map_loss_weight=${MAP_LOSS_WEIGHT}"
      echo "bayes_loss_weight=${BAYES_LOSS_WEIGHT}"
      echo "ctx_reg_weight=${CTX_REG_WEIGHT}"
      echo "use_full_cov=${USE_FULL_COV}"
      echo "model_selection=${MODEL_SELECTION}"
      echo "warmup_epoch=${WARMUP_EPOCH}"
      echo "warmup_cons_lr=${WARMUP_CONS_LR}"
      echo "cache_root=${CACHE_ROOT}"
      echo "rebuild_image_cache=${REBUILD_IMAGE_CACHE}"
      echo "disable_image_cache=${DISABLE_IMAGE_CACHE}"
      echo "=========================================="

      "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
        --dataset "${DATASET}" \
        --hessian_dir "${HESSIAN_DIR}" \
        --model "${MODEL_STR}" \
        --local_model_path "${MODEL_PATH}" \
        --data_root "${DATA_ROOT}" \
        --image_feature_cache_root "${CACHE_ROOT}" \
        --pseudo_data_count "${PSEUDO_DATA_COUNT}" \
        --lambda_txt_init "${LAMBDA_TXT_INIT}" \
        --lambda_opt_steps "${LAMBDA_OPT_STEPS}" \
        --n_ctx "${N_CTX}" \
        --ctx_init "${CTX_INIT}" \
        --class_token_position "${CLASS_TOKEN_POSITION}" \
        --shots_per_class "${SHOTS_PER_CLASS}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --warmup_epoch "${WARMUP_EPOCH}" \
        --warmup_cons_lr "${WARMUP_CONS_LR}" \
        --train_objective "${TRAIN_OBJECTIVE}" \
        --hybrid_warmup_epochs "${HYBRID_WARMUP_EPOCHS}" \
        --map_loss_weight "${MAP_LOSS_WEIGHT}" \
        --bayes_loss_weight "${BAYES_LOSS_WEIGHT}" \
        --ctx_reg_weight "${CTX_REG_WEIGHT}" \
        --model_selection "${MODEL_SELECTION}" \
        --save_dir "${OUTPUT_ROOT}" \
        --method_name "${METHOD_NAME}" \
        --prediction_topk "${PREDICTION_TOPK}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        "${EXTRA_ARGS[@]}"
    done
  done
done
