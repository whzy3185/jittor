#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
export nvcc_path="${nvcc_path-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
DATASET="${1:-dataset1}"
ARGS=(tune_blend.py --data-root data/official_raw --dataset "$DATASET")
if [[ -n "${CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${META:-}" ]]; then
  ARGS+=(--meta "$META")
fi
if [[ -n "${MAX_VAL_EVENTS:-}" ]]; then
  ARGS+=(--max-val-events "$MAX_VAL_EVENTS")
fi
if [[ -n "${MAX_TEST_QUERIES:-}" ]]; then
  ARGS+=(--max-test-queries "$MAX_TEST_QUERIES")
fi
if [[ -n "${WEIGHT_STEP:-}" ]]; then
  ARGS+=(--weight-step "$WEIGHT_STEP")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${WRITE_META:-}" ]]; then
  ARGS+=(--write-meta)
fi
if [[ -n "${USE_CUDA_ARGS:-}" ]]; then
  ARGS+=($USE_CUDA_ARGS)
fi
"$PYTHON_BIN" "${ARGS[@]}"
