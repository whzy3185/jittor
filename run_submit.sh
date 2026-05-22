#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
MODE="${JSON_MODE:-sorted_ids}"
DATASET="${1:-dataset1}"
ARGS=(track1_heuristic_submit.py --data-root data/official_raw --dataset "$DATASET" --output-root outputs/track1_heuristic_aggressive --json-mode "$MODE")
if [[ -n "${INCLUDE_VALID_HISTORY:-}" ]]; then
  ARGS+=(--include-valid-history)
fi
"$PYTHON_BIN" "${ARGS[@]}"
