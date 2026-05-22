#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
export nvcc_path="${nvcc_path-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
DATASET="${1:-dataset1}"
if [[ -z "${MAX_VAL_EVENTS:-}" ]]; then
  if [[ "$DATASET" == "dataset2" ]]; then
    MAX_VAL_EVENTS=5000
  else
    MAX_VAL_EVENTS=5000
  fi
fi
ARGS=(train.py --data-root data/official_raw --dataset "$DATASET" --output-dir "${OUTPUT_DIR:-outputs/track1}" --epochs "${EPOCHS:-30}" --batch-size "${BATCH_SIZE:-2048}" --negatives "${NEGATIVES:-5}" --val-negatives "${VAL_NEGATIVES:-50}")
ARGS+=(--emb-dim "${EMB_DIM:-128}" --hidden-dim "${HIDDEN_DIM:-256}" --time-dim "${TIME_DIM:-32}" --layers "${LAYERS:-3}" --dropout "${DROPOUT:-0.15}" --lr "${LR:-0.001}" --weight-decay "${WEIGHT_DECAY:-0.00001}" --seed "${SEED:-42}" --patience "${PATIENCE:-5}" --val-ratio "${VAL_RATIO:-0.1}")
if [[ -n "${MAX_TRAIN_EVENTS:-}" ]]; then
  ARGS+=(--max-train-events "$MAX_TRAIN_EVENTS")
fi
ARGS+=(--max-val-events "$MAX_VAL_EVENTS")
if [[ -n "${MAX_TEST_QUERIES:-}" ]]; then
  ARGS+=(--max-test-queries "$MAX_TEST_QUERIES")
fi
if [[ -n "${USE_CUDA_ARGS:-}" ]]; then
  ARGS+=($USE_CUDA_ARGS)
fi
"$PYTHON_BIN" "${ARGS[@]}"
