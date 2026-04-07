#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

METHOD_NAME="deterministic_coop_standard"
OUTPUT_ROOT="./output"
DATA_ROOT="./datasets"
MODEL_PATH="./models/clip-vit-b32"

CACHE_ROOT="./cache/image_features"

REBUILD_IMAGE_CACHE=0
DISABLE_IMAGE_CACHE=0

DATASETS=("food101")
SHOTS_PER_CLASS_LIST=(1)
SEEDS=(1)

N_CTX=16
CTX_INIT="a photo of a"
CSC=0
CLASS_TOKEN_POSITION="end"

LR=0.002
WEIGHT_DECAY=0
EPOCHS=100
BATCH_SIZE=256
NUM_WORKERS=8

PREDICTION_TOPK=5
DEVICE="cuda"
PYTHON_BIN="python"
TRAIN_SCRIPT="train_py/train_deterministic_coop.py"

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

for DATASET in "${DATASETS[@]}"; do
  for SHOTS_PER_CLASS in "${SHOTS_PER_CLASS_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "=========================================="
      echo "开始运行: method=${METHOD_NAME}, dataset=${DATASET}, shots=${SHOTS_PER_CLASS}, seed=${SEED}"
      echo "cache_root=${CACHE_ROOT}"
      echo "rebuild_image_cache=${REBUILD_IMAGE_CACHE}"
      echo "disable_image_cache=${DISABLE_IMAGE_CACHE}"
      echo "=========================================="

      "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
        --dataset "${DATASET}" \
        --model clip-base \
        --local_model_path "${MODEL_PATH}" \
        --data_root "${DATA_ROOT}" \
        --image_feature_cache_root "${CACHE_ROOT}" \
        --n_ctx "${N_CTX}" \
        --ctx_init "${CTX_INIT}" \
        --class_token_position "${CLASS_TOKEN_POSITION}" \
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