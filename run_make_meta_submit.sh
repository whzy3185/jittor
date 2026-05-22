#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"

"$PYTHON_BIN" make_testmeta_result_zip.py \
  --output-dir outputs/website_submission_testmeta_rank \
  --mode rank \
  --chunk-size "${CHUNK_SIZE:-5000}"

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_jittor_noprior_rank outputs/website_submission_testmeta_rank \
  --weights 0.65 0.18 0.07 0.10 \
  --output-dir outputs/website_submission_blend_enh65_hard18_jittor07_meta10 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_jittor_noprior_rank outputs/website_submission_testmeta_rank \
  --weights 0.60 0.15 0.05 0.20 \
  --output-dir outputs/website_submission_blend_enh60_hard15_jittor05_meta20 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_jittor_noprior_rank outputs/website_submission_order_prior_linear \
  --weights 0.68 0.18 0.09 0.05 \
  --output-dir outputs/website_submission_blend_enh68_hard18_jittor09_order05 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_jittor_noprior_rank outputs/website_submission_order_prior_linear \
  --weights 0.65 0.17 0.08 0.10 \
  --output-dir outputs/website_submission_blend_enh65_hard17_jittor08_order10 \
  --chunksize 8192

"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_enh65_hard18_jittor07_meta10/result.zip
"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_enh68_hard18_jittor09_order05/result.zip
