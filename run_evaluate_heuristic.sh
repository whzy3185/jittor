#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
DATASET="${1:-dataset1}"
ARGS=(evaluate_heuristic.py --data-root data/official_raw --dataset "$DATASET")
if [[ -n "${MAX_VAL_EVENTS:-}" ]]; then
  ARGS+=(--max-val-events "$MAX_VAL_EVENTS")
fi
if [[ -n "${MAX_TEST_QUERIES:-}" ]]; then
  ARGS+=(--max-test-queries "$MAX_TEST_QUERIES")
fi
if [[ -n "${VAL_NEGATIVES:-}" ]]; then
  ARGS+=(--val-negatives "$VAL_NEGATIVES")
fi
if [[ -n "${VAL_RATIO:-}" ]]; then
  ARGS+=(--val-ratio "$VAL_RATIO")
fi
if [[ -n "${WINDOW:-}" ]]; then
  ARGS+=(--window "$WINDOW")
fi
"$PYTHON_BIN" "${ARGS[@]}"
