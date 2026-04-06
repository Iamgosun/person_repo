#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

METHOD_NAME="deterministic_coop"
OUTPUT_ROOT="./output"
DATA_ROOT="./datasets"
MODEL_PATH="./models/clip-vit-b32"

# 图像特征缓存目录
CACHE_ROOT="./cache/image_features"

# 是否强制重建图像特征缓存：
# 0 = 直接复用已有缓存
# 1 = 删除命中并重新提取图像特征
REBUILD_IMAGE_CACHE=0

# 如需彻底关闭图像特征缓存，改成 1
DISABLE_IMAGE_CACHE=0

# 建议先用你当前最稳的 dataset 起跑
DATASETS=("food101")
SHOTS_PER_CLASS_LIST=(1)
SEEDS=(1)

# 做最干净的 deterministic CoOp 对照时，
# 若 ctx_init="a photo of a"，建议 n_ctx 先设成 4
N_CTX=3
CTX_INIT="a photo of"
FIXED_SUFFIX=", a type of food."




LR=1e-3
WEIGHT_DECAY=1e-4
EPOCHS=100
BATCH_SIZE=256
NUM_WORKERS=8

PREDICTION_TOPK=5
DEVICE="cuda"
PYTHON_BIN="python"
TRAIN_SCRIPT="train_py/train_deterministic_coop.py"

EXTRA_ARGS=()

# 图像缓存开关
if [[ "${REBUILD_IMAGE_CACHE}" -eq 1 ]]; then
  EXTRA_ARGS+=("--rebuild_image_feature_cache")
fi

if [[ "${DISABLE_IMAGE_CACHE}" -eq 1 ]]; then
  EXTRA_ARGS+=("--disable_cache_image_features")
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
        --fixed_suffix "${FIXED_SUFFIX}" \
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