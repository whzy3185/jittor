#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"

"$PYTHON_BIN" make_simple_jittor_result_zip.py \
  --model-dir outputs/simple_jittor_noprior \
  --output-dir outputs/website_submission_jittor_noprior_rank \
  --mode rank \
  --chunk-size "${CHUNK_SIZE:-5000}"

"$PYTHON_BIN" make_simple_jittor_result_zip.py \
  --model-dir outputs/simple_jittor_noprior \
  --output-dir outputs/website_submission_jittor_noprior_rowsigmoid \
  --mode row_sigmoid \
  --chunk-size "${CHUNK_SIZE:-5000}"

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_hard_shuffled outputs/website_submission_jittor_noprior_rank \
  --weights 0.7 0.3 \
  --output-dir outputs/website_submission_blend_hard70_jittor30 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_hard_shuffled outputs/website_submission_jittor_noprior_rank \
  --weights 0.5 0.5 \
  --output-dir outputs/website_submission_blend_hard50_jittor50 \
  --chunksize 8192

"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_hard70_jittor30/result.zip
"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_hard50_jittor50/result.zip
