#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLWELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

METHOD_NAME="text_only_bayes_coop"
OUTPUT_ROOT="./output"
DATA_ROOT="./datasets"
MODEL_PATH="./models/clip-vit-b32"
HESSIAN_DIR="./hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"

# 新增：图像特征缓存目录
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

# 第一次重构后排查建议保守一点
PSEUDO_DATA_COUNT=4
LAMBDA_TXT_INIT=300.0
LAMBDA_OPT_STEPS=100
N_CTX=16
CTX_INIT="a photo of a"

LR=1e-4
WEIGHT_DECAY=1e-5
EPOCHS=100
BATCH_SIZE=32

# 第一次先用 0，确认没问题后再改回 4
NUM_WORKERS=0

PREDICTION_TOPK=5
DEVICE="cuda"
PYTHON_BIN="python"
TRAIN_SCRIPT="train_py/train_text_only_bayes_coop.py"

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
        --hessian_dir "${HESSIAN_DIR}" \
        --model clip-base \
        --local_model_path "${MODEL_PATH}" \
        --data_root "${DATA_ROOT}" \
        --image_feature_cache_root "${CACHE_ROOT}" \
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