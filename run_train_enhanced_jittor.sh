#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
export nvcc_path="${nvcc_path:-}"
export cache_path="${cache_path:-/mnt/e/Jitter/.jittor_wsl_cpu_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/e/Jitter/.pip_cache}"

"$PYTHON_BIN" train_enhanced_jittor.py \
  --dataset dataset1 \
  --output-dir outputs/enhanced_jittor_mlp \
  --split-ratio "${SPLIT_RATIO:-0.85}" \
  --max-events "${MAX_EVENTS_DATASET1:-120000}" \
  --negatives "${NEGATIVES:-8}" \
  --epochs "${EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-4096}" \
  --hidden-dim "${HIDDEN_DIM:-64}" \
  --dropout "${DROPOUT:-0.08}" \
  --lr "${LR:-0.0015}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --seed "${SEED:-3031}" \
  --max-history-per-src "${MAX_HISTORY_PER_SRC:-30}" \
  --max-pairs-per-src "${MAX_PAIRS_PER_SRC:-30}"

"$PYTHON_BIN" train_enhanced_jittor.py \
  --dataset dataset2 \
  --output-dir outputs/enhanced_jittor_mlp \
  --split-ratio "${SPLIT_RATIO:-0.85}" \
  --max-events "${MAX_EVENTS_DATASET2:-120000}" \
  --negatives "${NEGATIVES:-8}" \
  --epochs "${EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-4096}" \
  --hidden-dim "${HIDDEN_DIM:-64}" \
  --dropout "${DROPOUT:-0.08}" \
  --lr "${LR:-0.0015}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --seed "${SEED:-3031}" \
  --max-history-per-src "${MAX_HISTORY_PER_SRC:-30}" \
  --max-pairs-per-src "${MAX_PAIRS_PER_SRC:-30}"
