#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DATASETS=("cifar10")
SHOTS_PER_CLASS_LIST=(1)
SEEDS=(1 2 3)

METHODS=(
  "LP:MEAN"
  "LP:RANDOM"
  "TR:TR"
  "ClipA:ClipA"
  "TipA:TipA"
  "CrossModal:CrossModal"
  "GAUSSIAN_PER_CLASS:GAUSSIAN_PER_CLASS"
)

MODEL="clip-base"
LOCAL_MODEL_PATH="./models/clip-vit-b32"
DATA_ROOT="./datasets"
SAVE_ROOT="./output_adapter"
METHOD_NAME="vlm_adapter"
PREDICTION_TOPK=5
DEVICE="cuda"
NUM_WORKERS=4
BATCH_SIZE=32
EPOCHS=20
LR=1e-3
WEIGHT_DECAY=1e-4

PYTHON_BIN="python"
TRAIN_SCRIPT="train_py/train_vlm_adapter.py"

EXTRA_ARGS=()

run_one() {
  local dataset="$1"
  local adapter_name="$2"
  local initialization="$3"
  local shots="$4"
  local seed="$5"

  echo "============================================================"
  echo "dataset=${dataset} method=${adapter_name} init=${initialization} shots=${shots} seed=${seed}"
  echo "save_root=${SAVE_ROOT}"
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