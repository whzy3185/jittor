from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from blend_adaptive_replay_gate import _copy_dataset1, _rank01, _row_margin, _source_features


def _write_dataset2(
    base_dir: Path,
    replay_dir: Path,
    stable_a_dir: Path,
    stable_b_dir: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    use_rank: bool,
) -> tuple[int, dict[str, float]]:
    total_src, seen_before, _, thresholds = _source_features(test_csv)
    paths = [
        base_dir / "dataset2.csv",
        replay_dir / "dataset2.csv",
        stable_a_dir / "dataset2.csv",
        stable_b_dir / "dataset2.csv",
    ]
    iters = [pd.read_csv(path, header=None, chunksize=chunksize) for path in paths]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    changed_top1 = 0
    bucket_counts: Counter[str] = Counter()
    offset = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunks in zip(*iters):
            base_mat, replay_mat, a_mat, b_mat = [chunk.to_numpy(np.float64) for chunk in chunks]
            out = np.empty_like(base_mat)
            for i in range(len(base_mat)):
                rid = offset + i
                base = base_mat[i]
                replay = replay_mat[i]
                stable_a = a_mat[i]
                stable_b = b_mat[i]
                b_top = int(np.argmax(base))
                r_top = int(np.argmax(replay))
                a_top = int(np.argmax(stable_a))
                c_top = int(np.argmax(stable_b))
                majority_top, majority_votes = Counter([b_top, a_top, c_top]).most_common(1)[0]
                hot = int(total_src[rid]) >= thresholds[2]
                warm = int(total_src[rid]) >= thresholds[1]
                active = int(seen_before[rid]) >= 8
                very_active = int(seen_before[rid]) >= 24
                replay_margin = _row_margin(replay)
                base_margin = _row_margin(base)

                if r_top == majority_top and majority_votes >= 2 and very_active:
                    chosen = replay
                    bucket = "hard_replay_majority_hot"
                elif r_top == majority_top and majority_votes >= 2 and active and hot:
                    chosen = 0.70 * replay + 0.30 * base
                    bucket = "mix_replay_majority_active_hot"
                elif r_top != b_top and r_top in (a_top, c_top) and hot and replay_margin >= 0.5 * max(base_margin, 1e-12):
                    chosen = 0.65 * replay + 0.35 * stable_b
                    bucket = "mix_replay_teacher_hot_disagree"
                elif b_top == majority_top and majority_votes >= 2 and r_top != b_top:
                    chosen = base
                    bucket = "hard_base_protect"
                elif not warm:
                    chosen = stable_b
                    bucket = "hard_stable_cold"
                elif r_top == b_top:
                    chosen = 0.50 * replay + 0.50 * base
                    bucket = "same_top_half"
                else:
                    chosen = 0.50 * base + 0.30 * stable_b + 0.20 * stable_a
                    bucket = "stable_mix_fallback"

                if use_rank:
                    if bucket.startswith("hard_replay"):
                        row = _rank01(replay)
                    elif bucket.startswith("hard_base"):
                        row = base
                    elif bucket.startswith("hard_stable"):
                        row = _rank01(stable_b)
                    else:
                        # Preserve the selected expert ordering while keeping a smooth row scale.
                        row = _rank01(chosen)
                else:
                    row = chosen
                out[i] = np.clip(row, 0.0, 1.0)
                changed_top1 += int(np.argmax(out[i]) != b_top)
                bucket_counts[bucket] += 1
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base_mat)
            offset += len(base_mat)
    stats = {
        "rows": float(rows),
        "top1_change_vs_base": float(changed_top1 / max(rows, 1)),
        "threshold_p50": float(thresholds[0]),
        "threshold_p75": float(thresholds[1]),
        "threshold_p90": float(thresholds[2]),
    }
    stats.update({f"bucket_{k}": float(v) for k, v in sorted(bucket_counts.items())})
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--stable-a-dir", required=True)
    parser.add_argument("--stable-b-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-space", action="store_true")
    parser.add_argument("--chunksize", type=int, default=4096)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _write_dataset2(
        base_dir,
        Path(args.replay_dir),
        Path(args.stable_a_dir),
        Path(args.stable_b_dir),
        Path(args.data_root) / "dataset2" / "test.csv",
        output_dir / "dataset2.csv",
        args.chunksize,
        use_rank=not args.score_space,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "stats": stats})


if __name__ == "__main__":
    main()
