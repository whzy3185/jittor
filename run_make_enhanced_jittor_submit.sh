#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
export nvcc_path="${nvcc_path:-}"
export cache_path="${cache_path:-/mnt/e/Jitter/.jittor_wsl_cpu_cache}"

"$PYTHON_BIN" make_enhanced_jittor_result_zip.py \
  --model-dir outputs/enhanced_jittor_mlp \
  --output-dir outputs/website_submission_enhanced_jittor_mlp_rank \
  --mode rank \
  --chunk-size "${CHUNK_SIZE:-5000}" \
  --max-history-per-src "${MAX_HISTORY_PER_SRC:-30}" \
  --max-pairs-per-src "${MAX_PAIRS_PER_SRC:-30}"

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_enhanced_jittor_mlp_rank \
  --weights 0.70 0.20 0.10 \
  --output-dir outputs/website_submission_blend_enh70_hard20_mlp10 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_enhanced_jittor_mlp_rank outputs/website_submission_testmeta_rank \
  --weights 0.60 0.15 0.15 0.10 \
  --output-dir outputs/website_submission_blend_enh60_hard15_mlp15_meta10 \
  --chunksize 8192

"$PYTHON_BIN" blend_website_csv.py \
  --inputs outputs/website_submission_enhanced_hard_rank outputs/website_submission_hard_shuffled outputs/website_submission_enhanced_jittor_mlp_rank outputs/website_submission_order_prior_linear \
  --weights 0.65 0.15 0.10 0.10 \
  --output-dir outputs/website_submission_blend_enh65_hard15_mlp10_order10 \
  --chunksize 8192

"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_enh70_hard20_mlp10/result.zip
"$PYTHON_BIN" validate_result_zip.py --zip outputs/website_submission_blend_enh60_hard15_mlp15_meta10/result.zip
