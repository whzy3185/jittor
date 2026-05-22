#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"

"$PYTHON_BIN" tune_sequential_hard.py \
  --dataset dataset1 \
  --output-dir outputs/sequential_heuristic_hard \
  --max-val-events "${MAX_VAL_EVENTS:-3000}" \
  --negatives "${NEGATIVES:-99}" \
  --trials "${TRIALS:-250}" \
  --seed "${SEED:-3032}" \
  --max-history-per-src "${MAX_HISTORY_PER_SRC:-30}" \
  --max-pairs-per-src "${MAX_PAIRS_PER_SRC:-30}"

"$PYTHON_BIN" tune_sequential_hard.py \
  --dataset dataset2 \
  --output-dir outputs/sequential_heuristic_hard \
  --max-val-events "${MAX_VAL_EVENTS:-3000}" \
  --negatives "${NEGATIVES:-99}" \
  --trials "${TRIALS:-250}" \
  --seed "${SEED:-3032}" \
  --max-history-per-src "${MAX_HISTORY_PER_SRC:-30}" \
  --max-pairs-per-src "${MAX_PAIRS_PER_SRC:-30}"
