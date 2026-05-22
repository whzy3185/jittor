from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from blend_adaptive_replay_gate import _copy_dataset1, _rank01
from track1_dynamic_rec.data import wide_candidate_columns


def _test_features(test_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header = pd.read_csv(test_csv, nrows=0)
    cand_cols = wide_candidate_columns(header.columns)
    srcs_all = []
    dup = []
    for chunk in pd.read_csv(test_csv, usecols=["src", *cand_cols], chunksize=8192):
        srcs = chunk["src"].to_numpy(np.int64)
        mat = chunk[cand_cols].to_numpy(np.int64)
        srcs_all.extend(int(x) for x in srcs)
        dup.extend(1 if len(set(map(int, row))) < len(row) else 0 for row in mat)
    counts = Counter(srcs_all)
    total = np.asarray([counts[int(x)] for x in srcs_all], dtype=np.int32)
    dup_arr = np.asarray(dup, dtype=np.int8)
    thresholds = np.quantile(np.asarray(list(counts.values()), dtype=np.float64), [0.5, 0.75, 0.9])
    return total, dup_arr, thresholds.astype(np.float64)


def _write_dataset2(
    base_dir: Path,
    meta_dir: Path,
    replay_gate_dir: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    mode: str,
) -> tuple[int, dict[str, float]]:
    total_src, dup_arr, thresholds = _test_features(test_csv)
    paths = [
        base_dir / "dataset2.csv",
        meta_dir / "dataset2.csv",
        replay_gate_dir / "dataset2.csv",
    ]
    iters = [pd.read_csv(path, header=None, chunksize=chunksize) for path in paths]
    rows = 0
    changed_top1 = 0
    bucket_counts: Counter[str] = Counter()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunks in zip(*iters):
            base_mat, meta_mat, replay_mat = [chunk.to_numpy(np.float64) for chunk in chunks]
            out = np.empty_like(base_mat)
            for i in range(len(base_mat)):
                rid = offset + i
                base = base_mat[i]
                meta = meta_mat[i]
                replay = replay_mat[i]
                base_top = int(np.argmax(base))
                meta_top = int(np.argmax(meta))
                replay_top = int(np.argmax(replay))
                hot = int(total_src[rid]) >= thresholds[2]
                warm = int(total_src[rid]) >= thresholds[1]
                duplicate = bool(dup_arr[rid])
                if mode == "bold":
                    if duplicate and meta_top != base_top:
                        row = 0.60 * _rank01(meta) + 0.40 * _rank01(replay)
                        bucket = "duplicate_meta_replay"
                    elif not warm and meta_top != base_top:
                        row = 0.55 * _rank01(meta) + 0.45 * base
                        bucket = "cold_meta"
                    elif hot and meta_top == replay_top:
                        row = 0.45 * _rank01(meta) + 0.35 * _rank01(replay) + 0.20 * base
                        bucket = "hot_meta_replay_agree"
                    elif meta_top == base_top:
                        row = 0.30 * _rank01(meta) + 0.70 * base
                        bucket = "same_top_meta_smooth"
                    else:
                        row = 0.70 * base + 0.20 * _rank01(meta) + 0.10 * _rank01(replay)
                        bucket = "base_protect"
                else:
                    if duplicate and meta_top == replay_top:
                        row = 0.40 * _rank01(meta) + 0.35 * _rank01(replay) + 0.25 * base
                        bucket = "duplicate_agree"
                    elif not warm and meta_top == replay_top:
                        row = 0.35 * _rank01(meta) + 0.65 * base
                        bucket = "cold_agree"
                    elif hot and meta_top == replay_top:
                        row = 0.30 * _rank01(meta) + 0.30 * _rank01(replay) + 0.40 * base
                        bucket = "hot_agree"
                    elif meta_top == base_top:
                        row = 0.20 * _rank01(meta) + 0.80 * base
                        bucket = "same_top"
                    else:
                        row = 0.88 * base + 0.08 * _rank01(meta) + 0.04 * _rank01(replay)
                        bucket = "protect"
                out[i] = np.clip(row, 0.0, 1.0)
                changed_top1 += int(np.argmax(out[i]) != base_top)
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
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--replay-gate-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["stable", "bold"], default="stable")
    parser.add_argument("--chunksize", type=int, default=4096)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _write_dataset2(
        base_dir,
        Path(args.meta_dir),
        Path(args.replay_gate_dir),
        Path(args.data_root) / "dataset2" / "test.csv",
        output_dir / "dataset2.csv",
        args.chunksize,
        args.mode,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "stats": stats})


if __name__ == "__main__":
    main()
