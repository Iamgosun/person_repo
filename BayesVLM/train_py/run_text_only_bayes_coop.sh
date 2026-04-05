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

DATASETS=("cifar10")
SHOTS_PER_CLASS_LIST=(16)
SEEDS=(1 2 3)

for DATASET in "${DATASETS[@]}"; do
  for SHOTS_PER_CLASS in "${SHOTS_PER_CLASS_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "=========================================="
      echo "开始运行: method=${METHOD_NAME}, dataset=${DATASET}, shots=${SHOTS_PER_CLASS}, seed=${SEED}"
      echo "=========================================="

      python -u train_py/train_text_only_bayes_coop.py \
        --dataset "${DATASET}" \
        --hessian_dir "${HESSIAN_DIR}" \
        --model clip-base \
        --local_model_path "${MODEL_PATH}" \
        --data_root "${DATA_ROOT}" \
        --pseudo_data_count 4 \
        --lambda_txt_init 300.0 \
        --lambda_opt_steps 1000 \
        --n_ctx 16 \
        --ctx_init "a photo of a" \
        --shots_per_class "${SHOTS_PER_CLASS}" \
        --lr 1e-4 \
        --weight_decay 1e-5 \
        --epochs 20 \
        --batch_size 256 \
        --num_workers 4 \
        --save_dir "${OUTPUT_ROOT}" \
        --method_name "${METHOD_NAME}" \
        --prediction_topk 5 \
        --seed "${SEED}" \
        --device cuda
    done
  done
done