#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
DATASET="${1:-dataset1}"
shift || true
if [[ "$#" -lt 1 ]]; then
  echo "usage: bash run_ensemble_scores.sh <dataset> <result_scores.csv> [more_result_scores.csv ...]" >&2
  exit 2
fi
ARGS=(ensemble_scores.py --data-root data/official_raw --dataset "$DATASET" --output-dir outputs/track1/ensemble_scores --scores "$@")
if [[ -n "${WEIGHTS:-}" ]]; then
  ARGS+=(--weights $WEIGHTS)
fi
if [[ -n "${MAX_QUERIES:-}" ]]; then
  ARGS+=(--max-queries "$MAX_QUERIES")
fi
if [[ -n "${CHUNK_SIZE:-}" ]]; then
  ARGS+=(--chunk-size "$CHUNK_SIZE")
fi
"$PYTHON_BIN" "${ARGS[@]}"
