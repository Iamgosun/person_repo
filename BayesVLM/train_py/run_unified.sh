#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ============================================================
# 统一训练脚本 run_unified.sh
# ------------------------------------------------------------
# 使用方式：
#   1. 直接在这个 .sh 文件里改参数
#   2. 保存后执行：
#        bash train_py/run_unified.sh
#
# 这份脚本统一管理三类任务：
#   - text_only_bayes_coop
#   - deterministic_coop
#   - vlm_adapter
#
# 你平时最常改的通常只有：
#   - RECIPE_NAME
#   - DATASETS
#   - SHOTS_PER_CLASS_LIST
#   - SEEDS
#   - METHODS                 # 仅 vlm_adapter 使用
#   - LR / EPOCHS / BATCH_SIZE
#   - MODEL_SELECTION
# ============================================================

# =========================
# 1) 任务类型
# -------------------------
# 可选值：
#   "text_only_bayes_coop"   -> 文本侧 Bayes CoOp
#   "deterministic_coop"     -> 标准 deterministic CoOp
#   "vlm_adapter"            -> 各类 adapter 方法
#
# 说明：
#   你一次只跑一个任务类型。
# =========================
RECIPE_NAME="vlm_adapter"
# RECIPE_NAME="text_only_bayes_coop"
# RECIPE_NAME="deterministic_coop"

# =========================
# 2) 基础路径
# -------------------------
# 说明：
#   这些一般只在换机器、换数据目录、换模型目录时改。
# =========================
OUTPUT_ROOT="./output"
DATA_ROOT="./datasets"
MODEL_PATH="./models/clip-vit-b32"
HESSIAN_DIR="./hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"
CACHE_ROOT="./cache/image_features"

# =========================
# 3) sweep 配置
# -------------------------
# 说明：
#   这里直接写死你要循环跑的实验组合。
#
# DATASETS:
#   你要跑的数据集列表
#
# SHOTS_PER_CLASS_LIST:
#   few-shot 设置，例如 1-shot / 2-shot / 4-shot / 8-shot / 16-shot
#
# SEEDS:
#   不同随机种子
#
# METHODS:
#   仅在 RECIPE_NAME="vlm_adapter" 时生效
#   格式必须写成：
#       "ADAPTER_NAME:INITIALIZATION"
#
#   你原仓库脚本里已经列出的可选 adapter 有：
#       LP
#       TR
#       CLIPA
#       TIPA
#       CROSSMODAL
#       GAUSSIAN_PER_CLASS
#       BAYESADAPTER
#
#   常见写法示例：
#       "LP:RANDOM"
#       "LP:MEAN"
#       "TR:TR"
#       "CLIPA:CLIPA"
#       "TIPA:TIPA"
#       "CROSSMODAL:CROSSMODAL"
#       "GAUSSIAN_PER_CLASS:GAUSSIAN_PER_CLASS"
#       "BAYESADAPTER:BAYESADAPTER"

# ========================= 
# "food101" "cifar10" "flowers102" "ucf101"
DATASETS=("ucf101" )

SHOTS_PER_CLASS_LIST=("1" "2" "4" "8" "16")
SEEDS=("1" ) # "1" "2" "3"
METHODS=(
  # "LP:RANDOM"
  # "LP:MEAN"
  # "TR:TR"
  # "CLIPA:CLIPA"
  # "TIPA:TIPA"
  # "CROSSMODAL:CROSSMODAL"
  # "GAUSSIAN_PER_CLASS:GAUSSIAN_PER_CLASS"
  "BAYESADAPTER:BAYESADAPTER"
)

# =========================
# 4) 全局训练配置
# -------------------------
# 说明：
#   这些参数三类任务基本都共用。
# =========================
MODEL_STR="clip-base"
DEVICE="cuda"
PYTHON_BIN="python"
TRAIN_SCRIPT="train_py/train_unified.py"

NUM_WORKERS=8
BATCH_SIZE=256
PREDICTION_TOPK=5

# =========================
# 5) 图像缓存相关
# -------------------------
# REBUILD_IMAGE_CACHE:
#   0 -> 复用已有缓存
#   1 -> 强制重建缓存
#
# DISABLE_IMAGE_CACHE:
#   0 -> 使用图像特征缓存
#   1 -> 不使用图像特征缓存
# =========================
REBUILD_IMAGE_CACHE=0
DISABLE_IMAGE_CACHE=0

USE_DATA_AUGMENTATION=0
USE_AUGMENTED_TRAIN_CACHE=0
TRAIN_AUG_REPEATS=20

# =========================
# 6) Prompt 相关通用参数
# -------------------------
# N_CTX:
#   prompt context token 数
#
# CTX_INIT:
#   prompt 初始化文本
#   空字符串表示不用文本初始化
#
# CSC:
#   0 -> 关闭 class-specific context
#   1 -> 开启 class-specific context
#
# CLASS_TOKEN_POSITION:
#   建议继续沿用原脚本默认值 "end"
# =========================
N_CTX=16
CTX_INIT=""
CSC=0
CLASS_TOKEN_POSITION="end"

# =========================
# 7) 优化器 / 调度器
# -------------------------
# OPTIMIZER 可选建议：
#   "sgd"
#   "adamw"
#   留空 -> 用不同任务各自默认值
#
# LR_SCHEDULER 可选建议：
#   "cosine"
#   "none"
#   留空 -> 用不同任务各自默认值
#
# MOMENTUM / NESTEROV:
#   主要对 SGD 有意义
# =========================
LR=0.1
WEIGHT_DECAY=0
EPOCHS=300

OPTIMIZER="sgd"
# OPTIMIZER="sgd"
# OPTIMIZER="adamw"

MOMENTUM=0.9
NESTEROV=0

LR_SCHEDULER="cosine"
# LR_SCHEDULER="cosine"
# LR_SCHEDULER="none"

WARMUP_EPOCH=1
WARMUP_CONS_LR=1e-5

# =========================
# 8) checkpoint 选择策略
# -------------------------
# MODEL_SELECTION:
#   "best" -> 用验证集最优轮
#   "last" -> 用最后一轮
#   留空   -> 用不同任务各自默认值
#
# SELECTION_METRIC:
#   "acc"  -> 按验证集准确率选 best
#   "loss" -> 按验证集 loss 选 best
#   "nlpd" -> 按验证集 nlpd 选 best
#   "ece"  -> 按验证集 ece 选 best
#   留空   -> 用不同任务各自默认值
#
# SELECTION_MODE:
#   "max"  -> 指标越大越好
#   "min"  -> 指标越小越好
#   "auto" -> 自动判断
#   留空   -> 用默认逻辑
#
# 推荐：
#   text_only_bayes_coop:
#       MODEL_SELECTION="last"
#
#   deterministic_coop:
#       MODEL_SELECTION="best"
#       SELECTION_METRIC="acc"
#       SELECTION_MODE="max"
#
#   vlm_adapter:
#       MODEL_SELECTION="best"
#       SELECTION_METRIC="loss"
#       SELECTION_MODE="min"
# =========================
MODEL_SELECTION="last"
# MODEL_SELECTION="best"
# MODEL_SELECTION="last"

SELECTION_METRIC="acc"
# SELECTION_METRIC="acc"
# SELECTION_METRIC="loss"
# SELECTION_METRIC="nlpd"
# SELECTION_METRIC="ece"

SELECTION_MODE="max"
# SELECTION_MODE="auto"
# SELECTION_MODE="min"
# SELECTION_MODE="max"

# =========================
# 9) text_only_bayes_coop 专属参数
# -------------------------
# TRAIN_OBJECTIVE 可选：
#   "map"
#   "bayes"
#   "hybrid"
#
# USE_FULL_COV:
#   0 -> 不用 full covariance
#   1 -> 使用 full covariance
# =========================
PSEUDO_DATA_COUNT=4
LAMBDA_TXT_INIT=300.0
LAMBDA_OPT_STEPS=1000

TRAIN_OBJECTIVE="bayes"
# TRAIN_OBJECTIVE="map"
# TRAIN_OBJECTIVE="hybrid"

HYBRID_WARMUP_EPOCHS=5
MAP_LOSS_WEIGHT=1.0
BAYES_LOSS_WEIGHT=1.0
CTX_REG_WEIGHT=1e-4

USE_FULL_COV=0

# =========================
# 10) vlm_adapter 专属参数
# -------------------------
# 说明：
#   这些参数只在对应 adapter 被 METHODS 选中时才有意义。
# =========================
TASKRES_ALPHA=0.5

CLIPA_RATIO=0.2
CLIPA_HIDDEN_DIM=0

TIPA_ALPHA=1.0
TIPA_BETA=1.0

GAUSSIAN_PRIOR_SIGMA=0.01
GAUSSIAN_MC_SAMPLES=3
GAUSSIAN_ANNEAL_START_EPOCH=20

BAYESADAPTER_PRIOR_SIGMA=0.01
BAYESADAPTER_TRAIN_MC_SAMPLES=3
BAYESADAPTER_EVAL_MC_SAMPLES=10
BAYESADAPTER_KL_SCALE_DIVISOR=1000.0

# =========================
# 11) 常用推荐组合
# -------------------------
# 1. 跑 text_only_bayes_coop
# RECIPE_NAME="text_only_bayes_coop"
# MODEL_SELECTION="last"
# TRAIN_OBJECTIVE="bayes"
#
# 2. 跑 deterministic_coop
# RECIPE_NAME="deterministic_coop"
# MODEL_SELECTION="best"
# SELECTION_METRIC="acc"
# SELECTION_MODE="max"
#
# 3. 跑 vlm_adapter
# RECIPE_NAME="vlm_adapter"
# MODEL_SELECTION="best"
# SELECTION_METRIC="loss"
# SELECTION_MODE="min"
# METHODS=("LP:RANDOM" "TR:TR" "TIPA:TIPA")
# =========================

# 允许在命令后追加额外 train_unified.py 参数
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

  if [[ "${USE_DATA_AUGMENTATION}" -eq 1 ]]; then
    EXTRA_ARGS+=("--use_data_augmentation")
  fi

  if [[ "${USE_AUGMENTED_TRAIN_CACHE}" -eq 1 ]]; then
    EXTRA_ARGS+=("--use_augmented_train_cache")
  fi

  EXTRA_ARGS+=("--train_aug_repeats" "${TRAIN_AUG_REPEATS}")
  
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
  echo "use_data_augmentation=${USE_DATA_AUGMENTATION}"
  echo "use_augmented_train_cache=${USE_AUGMENTED_TRAIN_CACHE}"
  echo "train_aug_repeats=${TRAIN_AUG_REPEATS}"
  echo "============================================================"
}

run_text_only_bayes_coop() {
  local dataset="$1"
  local shots="$2"
  local seed="$3"

  local method_name="text_only_bayes_coop_${TRAIN_OBJECTIVE}"

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

  # 默认保持与旧版 text_only_bayes_coop 习惯一致
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

  local method_name="deterministic_coop_standard"

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

  # 默认保持与 deterministic CoOp 习惯一致
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

  local method_name="vlm_adapter"

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

    --bayesadapter_prior_sigma "${BAYESADAPTER_PRIOR_SIGMA}"
    --bayesadapter_train_mc_samples "${BAYESADAPTER_TRAIN_MC_SAMPLES}"
    --bayesadapter_eval_mc_samples "${BAYESADAPTER_EVAL_MC_SAMPLES}"
    --bayesadapter_kl_scale_divisor "${BAYESADAPTER_KL_SCALE_DIVISOR}"

    --hessian_dir "${HESSIAN_DIR}"
    --pseudo_data_count "${PSEUDO_DATA_COUNT}"


  )

  # 默认保持与当前 vlm_adapter recipe 一致
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