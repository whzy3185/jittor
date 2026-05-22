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

make_geo() {
  local name="$1"
  local w1="$2"
  local w2="$3"
  local out="outputs/website_submission_${name}"
  echo "==> ${name}"
  "$PY" blend_website_csv_geomean.py \
    --inputs "$BASE" "$ONLINE" "$TEMP" "$SEQ" "$COMBO" "$ENH" \
    --weights-dataset1 "$w1" \
    --weights-dataset2 "$w2" \
    --output-dir "$out"
  mkdir -p "submissions/${name}"
  cp "$out/result.zip" "submissions/${name}/result.zip"
  "$PY" validate_result_zip.py --zip "submissions/${name}/result.zip"
}

# Input order: BASE, ONLINE, TEMP, SEQ, COMBO, ENH.
# Current verified best 56:
# dataset1 = 0.75 base + 0.20 online + 0.05 temp
# dataset2 = 0.05 online + 0.08 temp + 0.57 seq + 0.30 combo
make_linear "58_d1_56_d2_online03_combo32" "0.75,0.20,0.05,0,0,0" "0,0.03,0.08,0.57,0.32,0"
make_linear "59_d1_56_d2_online07_combo30" "0.75,0.20,0.05,0,0,0" "0,0.07,0.08,0.55,0.30,0"
make_linear "60_d1_56_d2_online07_combo32_temp06" "0.75,0.20,0.05,0,0,0" "0,0.07,0.06,0.55,0.32,0"
make_linear "61_d1_56_d2_online10_combo30_temp06" "0.75,0.20,0.05,0,0,0" "0,0.10,0.06,0.54,0.30,0"
make_linear "62_d1_56_d2_online05_combo34_temp06" "0.75,0.20,0.05,0,0,0" "0,0.05,0.06,0.55,0.34,0"
make_linear "63_d1_online18_d2_56" "0.77,0.18,0.05,0,0,0" "0,0.05,0.08,0.57,0.30,0"
make_linear "64_d1_online22_d2_56" "0.73,0.22,0.05,0,0,0" "0,0.05,0.08,0.57,0.30,0"
make_geo "65_geo_d1_56_d2_56" "0.75,0.20,0.05,0,0,0" "0,0.05,0.08,0.57,0.30,0"
