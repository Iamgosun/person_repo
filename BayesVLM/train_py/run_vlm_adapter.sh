#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# =========================
# sweep config
# =========================
DATASETS=("food101") #cifar10 food101
SHOTS_PER_CLASS_LIST=(16)
SEEDS=(1)

# 可选：
# "LP:MEAN"
# "LP:RANDOM"
# "TR:TR"
# "CLIPA:CLIPA"
# "TIPA:TIPA"
# "CROSSMODAL:CROSSMODAL"
# "GAUSSIAN_PER_CLASS:GAUSSIAN_PER_CLASS"
METHODS=(
  "LP:RANDOM"
)

# =========================
# global config
# =========================
MODEL="clip-base"
LOCAL_MODEL_PATH="./models/clip-vit-b32"
DATA_ROOT="./datasets"
SAVE_ROOT="./output"
METHOD_NAME="vlm_adapter"

# 新增：共享图像特征缓存
CACHE_ROOT="./cache/image_features"

# 0 = 复用图像缓存
# 1 = 强制重建图像缓存
REBUILD_IMAGE_CACHE=0

# 0 = 开启图像缓存
# 1 = 关闭图像缓存
DISABLE_IMAGE_CACHE=0

PREDICTION_TOPK=5
DEVICE="cuda"


NUM_WORKERS=8
BATCH_SIZE=256
EPOCHS=100
LR=1e-3
WEIGHT_DECAY=1e-4

# adapter extra config
TASKRES_ALPHA=0.5
CLIPA_RATIO=0.2
CLIPA_HIDDEN_DIM=0
TIPA_ALPHA=1.0
TIPA_BETA=1.0
GAUSSIAN_PRIOR_SIGMA=0.01
GAUSSIAN_MC_SAMPLES=3
GAUSSIAN_ANNEAL_START_EPOCH=20

PYTHON_BIN="python"
TRAIN_SCRIPT="train_py/train_vlm_adapter.py"

# 兼容旧接口保留，但 cached adapter 脚本本身不会直接用
HESSIAN_DIR="./hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"
PSEUDO_DATA_COUNT=4

# 如果后面想临时加额外参数，就往这里塞
EXTRA_ARGS=()

if [[ "${REBUILD_IMAGE_CACHE}" -eq 1 ]]; then
  EXTRA_ARGS+=("--rebuild_image_feature_cache")
fi

if [[ "${DISABLE_IMAGE_CACHE}" -eq 1 ]]; then
  EXTRA_ARGS+=("--disable_cache_image_features")
fi

run_one() {
  local dataset="$1"
  local adapter_name="$2"
  local initialization="$3"
  local shots="$4"
  local seed="$5"

  echo "============================================================"
  echo "dataset=${dataset} method=${adapter_name} init=${initialization} shots=${shots} seed=${seed}"
  echo "save_root=${SAVE_ROOT}"
  echo "cache_root=${CACHE_ROOT}"
  echo "rebuild_image_cache=${REBUILD_IMAGE_CACHE}"
  echo "disable_image_cache=${DISABLE_IMAGE_CACHE}"
  echo "============================================================"

  "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
    --dataset "${dataset}" \
    --model "${MODEL}" \
    --local_model_path "${LOCAL_MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --adapter_name "${adapter_name}" \
    --initialization "${initialization}" \
    --shots_per_class "${shots}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --save_dir "${SAVE_ROOT}" \
    --method_name "${METHOD_NAME}" \
    --prediction_topk "${PREDICTION_TOPK}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --taskres_alpha "${TASKRES_ALPHA}" \
    --clipa_ratio "${CLIPA_RATIO}" \
    --clipa_hidden_dim "${CLIPA_HIDDEN_DIM}" \
    --tipa_alpha "${TIPA_ALPHA}" \
    --tipa_beta "${TIPA_BETA}" \
    --gaussian_prior_sigma "${GAUSSIAN_PRIOR_SIGMA}" \
    --gaussian_mc_samples "${GAUSSIAN_MC_SAMPLES}" \
    --gaussian_anneal_start_epoch "${GAUSSIAN_ANNEAL_START_EPOCH}" \
    --image_feature_cache_root "${CACHE_ROOT}" \
    --hessian_dir "${HESSIAN_DIR}" \
    --pseudo_data_count "${PSEUDO_DATA_COUNT}" \
    "${EXTRA_ARGS[@]}"
}

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    IFS=':' read -r adapter_name initialization <<< "${method}"
    for shots in "${SHOTS_PER_CLASS_LIST[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one "${dataset}" "${adapter_name}" "${initialization}" "${shots}" "${seed}"
      done
    done
  done
done

echo "All runs finished."