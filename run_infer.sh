#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
export nvcc_path="${nvcc_path-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
DATASET="${1:-dataset1}"
ARGS=(infer.py --data-root data/official_raw --dataset "$DATASET" --output-dir outputs/track1/submission)
if [[ -n "${MAX_VAL_EVENTS:-}" ]]; then
  ARGS+=(--max-val-events "$MAX_VAL_EVENTS")
fi
if [[ -n "${MAX_TEST_QUERIES:-}" ]]; then
  ARGS+=(--max-test-queries "$MAX_TEST_QUERIES")
fi
if [[ -n "${USE_CUDA_ARGS:-}" ]]; then
  ARGS+=($USE_CUDA_ARGS)
fi
"$PYTHON_BIN" "${ARGS[@]}"
