#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"

"$PYTHON_BIN" blend_website_csv_by_dataset.py \
  --inputs outputs/website_submission_sequential_hard_rank outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_enhanced_jittor_mlp_rank \
  --weights-dataset1 0.55,0.30,0.10,0.05 \
  --weights-dataset2 0.80,0.08,0.07,0.05 \
  --output-dir outputs/website_submission_blend_dsaware_seq \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv_by_dataset.py \
  --inputs outputs/website_submission_sequential_hard_rank outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_testmeta_rank \
  --weights-dataset1 0.50,0.30,0.10,0.10 \
  --weights-dataset2 0.75,0.08,0.07,0.10 \
  --output-dir outputs/website_submission_blend_dsaware_seq_meta \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv_by_dataset.py \
  --inputs outputs/website_submission_sequential_hard_rank outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_order_prior_linear \
  --weights-dataset1 0.50,0.32,0.10,0.08 \
  --weights-dataset2 0.72,0.08,0.07,0.13 \
  --output-dir outputs/website_submission_blend_dsaware_seq_order \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv_by_dataset.py \
  --inputs outputs/website_submission_sequential_hard_rank outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_enhanced_jittor_mlp_rank \
  --weights-dataset1 0.55,0.30,0.10,0.05 \
  --weights-dataset2 0.80,0.08,0.07,0.05 \
  --shrink-dataset1 0.05 \
  --shrink-dataset2 0.08 \
  --output-dir outputs/website_submission_blend_dsaware_seq_shrink \
  --chunksize 8192

"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_dsaware_seq/result.zip
"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_dsaware_seq_shrink/result.zip
