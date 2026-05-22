#!/usr/bin/env bash
set -euo pipefail

cd /mnt/e/Jitter/track1_aggressive_wsl

keep_dir() {
  case "$1" in
    website_submission_blend_dsaware_seq) return 0 ;;
    website_submission_sequential_fullhistory_rank) return 0 ;;
    website_submission_online_seq_fullhistory_q50_rank) return 0 ;;
    website_submission_temporal_tuned_fullhistory_rank) return 0 ;;
    website_submission_enhanced_fullhistory_rank) return 0 ;;
    website_submission_groupmeta_rank) return 0 ;;
    website_submission_assoc_tuned_fullhistory_rank) return 0 ;;
    website_submission_combo_jittor_fullhistory_rank) return 0 ;;
    *) return 1 ;;
  esac
}

echo "Before cleanup:"
du -sh outputs submissions || true

echo "Deleting redundant website_submission* output directories:"
for d in outputs/website_submission*; do
  [ -d "$d" ] || continue
  base="${d##*/}"
  if keep_dir "$base"; then
    echo "KEEP $d"
  else
    echo "DELETE $d"
    rm -rf -- "$d"
  fi
done

echo "Deleting smoke/intermediate output directories:"
find outputs -maxdepth 1 -type d \( -name "smoke*" -o -name "*smoke*" \) -print -exec rm -rf -- {} +

if [ -d outputs/track1_heuristic_aggressive ]; then
  echo "DELETE outputs/track1_heuristic_aggressive"
  rm -rf -- outputs/track1_heuristic_aggressive
fi

echo "After cleanup:"
du -sh outputs submissions || true

echo "Remaining website_submission* output directories:"
find outputs -maxdepth 1 -type d -name "website_submission*" | sort
