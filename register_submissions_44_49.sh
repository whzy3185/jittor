#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/Jitter/track1_aggressive_wsl

items=(
  "44_d1_more_online_d2_43:d1_more_online_d2_43"
  "45_d1_less_online_d2_43:d1_less_online_d2_43"
  "46_d1_43_d2_combo28:d1_43_d2_combo28"
  "47_d1_43_d2_combo18:d1_43_d2_combo18"
  "48_d1_43_d2_online10:d1_43_d2_online10"
  "49_d1_43_d2_combo26_enh08:d1_43_d2_combo26_enh08"
)

for item in "${items[@]}"; do
  sub="${item%%:*}"
  out="${item##*:}"
  echo "VALIDATE outputs/website_submission_${out}/result.zip"
  /mnt/e/Jitter/.venv_wsl_cpu/bin/python validate_result_zip.py --zip "outputs/website_submission_${out}/result.zip"
  mkdir -p "submissions/${sub}"
  cp "outputs/website_submission_${out}/result.zip" "submissions/${sub}/result.zip"
  echo "CHECK submissions/${sub}/result.zip"
  /mnt/e/Jitter/.venv_wsl_cpu/bin/python validate_result_zip.py --zip "submissions/${sub}/result.zip"
done

find submissions -maxdepth 2 -type f -name result.zip -printf '%p %s bytes\n' | sort | tail -12
