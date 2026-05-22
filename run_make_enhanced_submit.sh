#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"

"$PYTHON_BIN" make_enhanced_result_zip.py \
  --model-dir outputs/enhanced_heuristic_hard \
  --output-dir outputs/website_submission_enhanced_hard_rank \
  --mode rank \
  --chunk-size "${CHUNK_SIZE:-5000}" \
  --max-history-per-src "${MAX_HISTORY_PER_SRC:-30}" \
  --max-pairs-per-src "${MAX_PAIRS_PER_SRC:-30}"

"$PYTHON_BIN" make_enhanced_result_zip.py \
  --model-dir outputs/enhanced_heuristic_hard \
  --output-dir outputs/website_submission_enhanced_hard_rowsigmoid \
  --mode row_sigmoid \
  --chunk-size "${CHUNK_SIZE:-5000}" \
  --max-history-per-src "${MAX_HISTORY_PER_SRC:-30}" \
  --max-pairs-per-src "${MAX_PAIRS_PER_SRC:-30}"

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_jittor_noprior_rank \
  --weights 0.7 0.2 0.1 \
  --output-dir outputs/website_submission_blend_enh70_hard20_jittor10 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled \
  --weights 0.8 0.2 \
  --output-dir outputs/website_submission_blend_enh80_hard20 \
  --chunksize 8192

"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_enhanced_hard_rank/result.zip
"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_enh70_hard20_jittor10/result.zip
"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_enh80_hard20/result.zip
