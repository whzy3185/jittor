#!/usr/bin/env bash
set -euo pipefail
export PIP_CACHE_DIR=/mnt/e/Jitter/.pip_cache
export cache_path=/mnt/e/Jitter/.jittor_wsl_cpu_cache
export nvcc_path="${nvcc_path-}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/e/Jitter/.venv_wsl_cpu/bin/python}"

echo "== WSL"
uname -a
pwd
whoami
python3 --version || python --version
nvidia-smi || true

echo "== E drive"
ls -lah /mnt/e | sed -n '1,12p'

echo "== paths"
echo "baseline=/mnt/e/Jitter/legacy_track1"
echo "dataset=/mnt/e/Jitter/data/official_raw"
echo "workdir=/mnt/e/Jitter/track1_aggressive_wsl"
test -d /mnt/e/Jitter/legacy_track1
test -d /mnt/e/Jitter/data/official_raw/dataset1
test -d /mnt/e/Jitter/data/official_raw/dataset2

echo "== imports"
"$PYTHON_BIN" - <<'PY'
import jittor as jt
import jittor_geometric
print("jittor", getattr(jt, "__version__", "unknown"), "cuda", bool(jt.flags.use_cuda))
print("jittor_geometric", "ok")
PY

echo "== python syntax"
"$PYTHON_BIN" -m py_compile \
  train.py infer.py infer_stream.py ensemble_scores.py tune_blend.py evaluate_heuristic.py \
  validate_submission.py package_release.py track1_heuristic_submit.py track1_dynamic_rec/*.py

echo "== no torch imports"
if find . -path './outputs' -prune -o -path './__pycache__' -prune -o -name '*.py' -type f -print | xargs grep -n 'torch'; then
  echo "unexpected torch reference found" >&2
  exit 1
else
  echo "NO_TORCH_IN_PY"
fi

echo "== docs"
test -f README_CN.md
test -f METHOD_CN.md
ls -lh README_CN.md METHOD_CN.md

echo "== submissions"
"$PYTHON_BIN" validate_submission.py --data-root data/official_raw --dataset dataset1 --submission outputs/track1_heuristic_aggressive/dataset1.zip
"$PYTHON_BIN" validate_submission.py --data-root data/official_raw --dataset dataset2 --submission outputs/track1_heuristic_aggressive/dataset2.zip
ls -lh outputs/track1_heuristic_aggressive/dataset1.zip outputs/track1_heuristic_aggressive/dataset2.zip

echo "== release"
ls -lh outputs/release || true

echo "== tree"
find . -maxdepth 2 -type f \( -name '*.py' -o -name '*.sh' -o -name 'README_CN.md' -o -name 'METHOD_CN.md' \) | sort
