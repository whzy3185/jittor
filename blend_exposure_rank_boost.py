from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _copy_dataset1(base_dir: Path, output_dir: Path, chunksize: int) -> int:
    rows = 0
    with (output_dir / "dataset1.csv").open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunk in pd.read_csv(base_dir / "dataset1.csv", header=None, chunksize=chunksize):
            for row in chunk.to_numpy(np.float64):
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(chunk)
    return rows


def _causal_src_seen(test_csv: Path) -> np.ndarray:
    test = pd.read_csv(test_csv, usecols=["src", "time"])
    test = test.assign(_row_id=test.index.to_numpy(np.int64)).sort_values(["time", "_row_id"]).reset_index(drop=True)
    seen_counts = np.zeros(len(test), dtype=np.int32)
    seen: Counter[int] = Counter()
    for _, group in test.groupby("time", sort=False):
        srcs = group["src"].to_numpy(np.int64)
        row_ids = group["_row_id"].to_numpy(np.int64)
        for src, row_id in zip(srcs, row_ids):
            seen_counts[int(row_id)] = int(seen[int(src)])
        seen.update(int(x) for x in srcs)
    return seen_counts


def _row_rank_boost(
    base: np.ndarray,
    expert: np.ndarray,
    seen_count: int,
    *,
    warm_seen: int,
    hot_seen: int,
    warm_alpha: float,
    hot_alpha: float,
    low_agree_scale: float,
    topn: int,
    min_top_overlap: int,
) -> tuple[np.ndarray, bool, bool]:
    if seen_count < warm_seen:
        return base, False, False
    alpha = hot_alpha if seen_count >= hot_seen else warm_alpha
    base_order = np.argsort(-base)
    expert_order = np.argsort(-expert)
    overlap = len(set(base_order[:topn]).intersection(int(x) for x in expert_order[:topn]))
    agreed = overlap >= min_top_overlap
    if not agreed:
        alpha *= low_agree_scale
    if alpha <= 0.0:
        return base, False, agreed

    adjusted = base + alpha * (expert - 0.5)
    new_order = np.argsort(-adjusted)

    # Preserve the exact row-wise score multiset from the strong base submission.
    out = np.empty_like(base)
    out[new_order] = np.sort(base)[::-1]
    changed = not np.array_equal(new_order, base_order)
    return out, changed, agreed


def _blend_dataset2(
    base_csv: Path,
    expert_csv: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    warm_seen: int,
    hot_seen: int,
    warm_alpha: float,
    hot_alpha: float,
    low_agree_scale: float,
    topn: int,
    min_top_overlap: int,
) -> tuple[int, dict[str, float]]:
    seen_counts = _causal_src_seen(test_csv)
    rows = 0
    changed_rows = 0
    agree_rows = 0
    warm_rows = 0
    hot_rows = 0
    offset = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
        expert_iter = pd.read_csv(expert_csv, header=None, chunksize=chunksize)
        for base_chunk, expert_chunk in zip(base_iter, expert_iter):
            base_mat = base_chunk.to_numpy(np.float64)
            expert_mat = expert_chunk.to_numpy(np.float64)
            out = np.empty_like(base_mat)
            for i in range(len(base_mat)):
                seen = int(seen_counts[offset + i])
                if seen >= hot_seen:
                    hot_rows += 1
                elif seen >= warm_seen:
                    warm_rows += 1
                row, changed, agreed = _row_rank_boost(
                    base_mat[i],
                    expert_mat[i],
                    seen,
                    warm_seen=warm_seen,
                    hot_seen=hot_seen,
                    warm_alpha=warm_alpha,
                    hot_alpha=hot_alpha,
                    low_agree_scale=low_agree_scale,
                    topn=topn,
                    min_top_overlap=min_top_overlap,
                )
                out[i] = row
                changed_rows += int(changed)
                agree_rows += int(agreed)
            for row in np.clip(out, 0.0, 1.0):
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base_mat)
            offset += len(base_mat)
    return rows, {
        "changed_rows": float(changed_rows),
        "agree_rows": float(agree_rows),
        "warm_rows": float(warm_rows),
        "hot_rows": float(hot_rows),
        "warm_seen": float(warm_seen),
        "hot_seen": float(hot_seen),
        "warm_alpha": float(warm_alpha),
        "hot_alpha": float(hot_alpha),
        "low_agree_scale": float(low_agree_scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warm-seen", type=int, default=8)
    parser.add_argument("--hot-seen", type=int, default=40)
    parser.add_argument("--warm-alpha", type=float, default=0.10)
    parser.add_argument("--hot-alpha", type=float, default=0.24)
    parser.add_argument("--low-agree-scale", type=float, default=0.25)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--min-top-overlap", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=8192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _blend_dataset2(
        base_dir / "dataset2.csv",
        Path(args.expert_dir) / "dataset2.csv",
        Path(args.data_root) / "dataset2" / "test.csv",
        output_dir / "dataset2.csv",
        args.chunksize,
        args.warm_seen,
        args.hot_seen,
        args.warm_alpha,
        args.hot_alpha,
        args.low_agree_scale,
        args.topn,
        args.min_top_overlap,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "stats": stats})


if __name__ == "__main__":
    main()
