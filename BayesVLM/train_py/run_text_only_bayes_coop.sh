#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

METHOD_NAME="text_only_bayes_coop"
OUTPUT_ROOT="./output"
DATA_ROOT="./datasets"
MODEL_PATH="./models/clip-vit-b32"
HESSIAN_DIR="./hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"

# 建议先用你当前最稳的 dataset 起跑
DATASETS=("cifar10")
SHOTS_PER_CLASS_LIST=(1)
SEEDS=(1)

# 第一次重构后排查建议保守一点
PSEUDO_DATA_COUNT=4
LAMBDA_TXT_INIT=300.0
LAMBDA_OPT_STEPS=100
N_CTX=16
CTX_INIT="a photo of a"

LR=1e-4
WEIGHT_DECAY=1e-5
EPOCHS=5
BATCH_SIZE=32

# 第一次先用 0，确认没问题后再改回 4
NUM_WORKERS=0

PREDICTION_TOPK=5
DEVICE="cuda"
PYTHON_BIN="python"
TRAIN_SCRIPT="train_py/train_text_only_bayes_coop.py"

EXTRA_ARGS=(
)

for DATASET in "${DATASETS[@]}"; do
  for SHOTS_PER_CLASS in "${SHOTS_PER_CLASS_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "=========================================="
      echo "开始运行: method=${METHOD_NAME}, dataset=${DATASET}, shots=${SHOTS_PER_CLASS}, seed=${SEED}"
      echo "=========================================="

      "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
        --dataset "${DATASET}" \
        --hessian_dir "${HESSIAN_DIR}" \
        --model clip-base \
        --local_model_path "${MODEL_PATH}" \
        --data_root "${DATA_ROOT}" \
        --pseudo_data_count "${PSEUDO_DATA_COUNT}" \
        --lambda_txt_init "${LAMBDA_TXT_INIT}" \
        --lambda_opt_steps "${LAMBDA_OPT_STEPS}" \
        --n_ctx "${N_CTX}" \
        --ctx_init "${CTX_INIT}" \
        --shots_per_class "${SHOTS_PER_CLASS}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --save_dir "${OUTPUT_ROOT}" \
        --method_name "${METHOD_NAME}" \
        --prediction_topk "${PREDICTION_TOPK}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        "${EXTRA_ARGS[@]}"
    done
  done
done