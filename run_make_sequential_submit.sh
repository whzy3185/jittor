#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"

"$PYTHON_BIN" make_sequential_result_zip.py \
  --model-dir outputs/sequential_heuristic_hard \
  --output-dir outputs/website_submission_sequential_hard_rank \
  --mode rank \
  --chunk-size "${CHUNK_SIZE:-5000}" \
  --max-history-per-src "${MAX_HISTORY_PER_SRC:-30}" \
  --max-pairs-per-src "${MAX_PAIRS_PER_SRC:-30}"

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_sequential_hard_rank outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_enhanced_jittor_mlp_rank \
  --weights 0.70 0.15 0.10 0.05 \
  --output-dir outputs/website_submission_blend_seq70_enh15_hard10_mlp05 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_sequential_hard_rank outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_testmeta_rank \
  --weights 0.65 0.15 0.10 0.10 \
  --output-dir outputs/website_submission_blend_seq65_enh15_hard10_meta10 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_sequential_hard_rank outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_order_prior_linear \
  --weights 0.65 0.15 0.10 0.10 \
  --output-dir outputs/website_submission_blend_seq65_enh15_hard10_order10 \
  --chunksize 8192

"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_seq70_enh15_hard10_mlp05/result.zip
"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_seq65_enh15_hard10_meta10/result.zip
