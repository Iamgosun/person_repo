#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
MODEL_DIR="./models/clip-vit-b32"


echo "[2/5] Prepare environment..."
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60

rm -rf "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

echo "[3/5] Download required files only..."
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    local_dir="./models/clip-vit-b32",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
    ],
    max_workers=1,
)
print("download finished")
PY

echo "[4/5] Check downloaded files..."
ls -lh "${MODEL_DIR}"

echo "[5/5] Validate local loading..."
python - <<'PY'
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

path = "./models/clip-vit-b32"

AutoTokenizer.from_pretrained(path, local_files_only=True)
CLIPTextModelWithProjection.from_pretrained(path, local_files_only=True)
CLIPVisionModelWithProjection.from_pretrained(path, local_files_only=True)
CLIPModel.from_pretrained(path, local_files_only=True)

print("all ok")
PY

echo "[DONE] Model is ready at: ${MODEL_DIR}"