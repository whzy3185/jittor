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
# Current verified best 43 is approximately:
# dataset1 = 0.80 base + 0.15 online + 0.05 temp
# dataset2 = 0.68 seq + 0.10 temp + 0.22 combo
make_linear "50_d1_online18_d2_combo24" "0.77,0.18,0.05,0,0,0" "0,0,0.10,0.66,0.24,0"
make_linear "51_d1_online22_d2_combo24" "0.73,0.22,0.05,0,0,0" "0,0,0.10,0.66,0.24,0"
make_linear "52_d1_online18_d2_combo28" "0.77,0.18,0.05,0,0,0" "0,0,0.10,0.62,0.28,0"
make_linear "53_d1_online22_d2_combo28" "0.73,0.22,0.05,0,0,0" "0,0,0.10,0.62,0.28,0"
make_linear "54_d1_online18_d2_combo32" "0.77,0.18,0.05,0,0,0" "0,0,0.10,0.58,0.32,0"
make_linear "55_d1_online20_d2_combo30_temp06" "0.75,0.20,0.05,0,0,0" "0,0,0.06,0.64,0.30,0"
make_linear "56_d1_online20_d2_combo30_online05" "0.75,0.20,0.05,0,0,0" "0,0.05,0.08,0.57,0.30,0"
make_geo "57_geo_d1_online18_d2_combo28" "0.77,0.18,0.05,0,0,0" "0,0,0.10,0.62,0.28,0"
