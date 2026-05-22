#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/e/Jitter/track1_aggressive_wsl"
SUB="$ROOT/submissions"
LOG="$SUB/deleted_obsolete_submissions_20260512.txt"

cd "$ROOT"

resolved_sub="$(readlink -f "$SUB")"
if [[ "$resolved_sub" != "/mnt/e/Jitter/track1_aggressive_wsl/submissions" ]]; then
  echo "Refuse: bad submissions path $resolved_sub" >&2
  exit 1
fi

: > "$LOG"

for d in submissions/{02..39}_*; do
  [[ -d "$d" ]] || continue
  target="$(readlink -f "$d")"
  case "$target" in
    /mnt/e/Jitter/track1_aggressive_wsl/submissions/*) ;;
    *)
      echo "Refuse: bad target $target" >&2
      exit 1
      ;;
  esac
  du -sh "$d" >> "$LOG"
  rm -rf -- "$d"
done

echo "deleted obsolete submissions 02-39; kept 01, 40, 41, 42, 43, 44-57" >> "$LOG"
cat "$LOG"
