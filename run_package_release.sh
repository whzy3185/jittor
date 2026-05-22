#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"
"$PYTHON_BIN" validate_submission.py --data-root data/official_raw --dataset dataset1 --submission outputs/track1_heuristic_aggressive/dataset1.zip
"$PYTHON_BIN" validate_submission.py --data-root data/official_raw --dataset dataset2 --submission outputs/track1_heuristic_aggressive/dataset2.zip
"$PYTHON_BIN" package_release.py --output-dir "${OUTPUT_DIR:-outputs/release}"
