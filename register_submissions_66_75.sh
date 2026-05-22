#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/e/Jitter/track1_aggressive_wsl"
PY="/mnt/e/Jitter/.venv_wsl_cpu/bin/python"
cd "$ROOT"

BASE="outputs/website_submission_blend_dsaware_seq"
ONLINE="outputs/website_submission_online_seq_fullhistory_q50_rank"
TEMP="outputs/website_submission_temporal_tuned_fullhistory_rank"
SEQ="outputs/website_submission_sequential_fullhistory_rank"
COMBO="outputs/website_submission_combo_jittor_fullhistory_rank"
ENH="outputs/website_submission_enhanced_fullhistory_rank"
S62="outputs/website_submission_62_d1_56_d2_online05_combo34_temp06"
S65="outputs/website_submission_65_geo_d1_56_d2_56"
S57="outputs/website_submission_57_geo_d1_online18_d2_combo28"

make_linear() {
  local name="$1"
  local w1="$2"
  local w2="$3"
  local out="outputs/website_submission_${name}"
  echo "==> ${name}"
  "$PY" blend_website_csv_by_dataset.py \
    --inputs "$BASE" "$ONLINE" "$TEMP" "$SEQ" "$COMBO" "$ENH" \
    --weights-dataset1 "$w1" \
    --weights-dataset2 "$w2" \
    --output-dir "$out"
  mkdir -p "submissions/${name}"
  cp "$out/result.zip" "submissions/${name}/result.zip"
  "$PY" validate_result_zip.py --zip "submissions/${name}/result.zip"
}

make_existing_blend() {
  local name="$1"
  local inputs="$2"
  local weights="$3"
  local out="outputs/website_submission_${name}"
  echo "==> ${name}"
  # shellcheck disable=SC2086
  "$PY" blend_website_csv_by_dataset.py \
    --inputs $inputs \
    --weights-dataset1 "$weights" \
    --weights-dataset2 "$weights" \
    --output-dir "$out"
  mkdir -p "submissions/${name}"
  cp "$out/result.zip" "submissions/${name}/result.zip"
  "$PY" validate_result_zip.py --zip "submissions/${name}/result.zip"
}

# Input order: BASE, ONLINE, TEMP, SEQ, COMBO, ENH.
# Current verified best 62:
# dataset1 = 0.75 base + 0.20 online + 0.05 temp
# dataset2 = 0.05 online + 0.06 temp + 0.55 seq + 0.34 combo
make_linear "66_d1_62_d2_combo36_temp06" "0.75,0.20,0.05,0,0,0" "0,0.05,0.06,0.53,0.36,0"
make_linear "67_d1_62_d2_combo36_temp04" "0.75,0.20,0.05,0,0,0" "0,0.05,0.04,0.55,0.36,0"
make_linear "68_d1_62_d2_online03_combo36_temp06" "0.75,0.20,0.05,0,0,0" "0,0.03,0.06,0.55,0.36,0"
make_linear "69_d1_62_d2_online07_combo36_temp06" "0.75,0.20,0.05,0,0,0" "0,0.07,0.06,0.51,0.36,0"
make_linear "70_d1_62_d2_combo38_temp04" "0.75,0.20,0.05,0,0,0" "0,0.05,0.04,0.53,0.38,0"
make_linear "71_d1_62_d2_online03_combo38_temp04" "0.75,0.20,0.05,0,0,0" "0,0.03,0.04,0.55,0.38,0"
make_linear "72_d1_online22_d2_62" "0.73,0.22,0.05,0,0,0" "0,0.05,0.06,0.55,0.34,0"
make_linear "73_d1_online18_d2_62" "0.77,0.18,0.05,0,0,0" "0,0.05,0.06,0.55,0.34,0"

# Small linear blends with the useful-but-weaker geomean runs.
make_existing_blend "74_s62_95_geo65_05" "$S62 $S65" "0.95,0.05"
make_existing_blend "75_s62_97_geo57_03" "$S62 $S57" "0.97,0.03"
