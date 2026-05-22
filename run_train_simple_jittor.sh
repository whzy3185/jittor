#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
export nvcc_path="${nvcc_path:-}"
export cache_path="${cache_path:-/mnt/e/Jitter/.jittor_wsl_cpu_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/e/Jitter/.pip_cache}"

"$PYTHON_BIN" train_simple_jittor.py \
  --dataset dataset1 \
  --output-dir outputs/simple_jittor_noprior \
  --max-events "${MAX_EVENTS_DATASET1:-200000}" \
  --warmup-events "${WARMUP_EVENTS:-10000}" \
  --negatives "${NEGATIVES:-8}" \
  --epochs "${EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-4096}" \
  --lr "${LR:-0.03}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --seed "${SEED:-2027}"

"$PYTHON_BIN" train_simple_jittor.py \
  --dataset dataset2 \
  --output-dir outputs/simple_jittor_noprior \
  --max-events "${MAX_EVENTS_DATASET2:-200000}" \
  --warmup-events "${WARMUP_EVENTS:-10000}" \
  --negatives "${NEGATIVES:-8}" \
  --epochs "${EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-4096}" \
  --lr "${LR:-0.03}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --seed "${SEED:-2027}"
