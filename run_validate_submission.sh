#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
DATASET="${1:-dataset1}"
SUBMISSION="${2:-outputs/track1_heuristic_aggressive/${DATASET}.zip}"
ARGS=(validate_submission.py --data-root data/official_raw --dataset "$DATASET" --submission "$SUBMISSION")
if [[ -n "${MAX_QUERIES:-}" ]]; then
  ARGS+=(--max-queries "$MAX_QUERIES")
fi
"$PYTHON_BIN" "${ARGS[@]}"
