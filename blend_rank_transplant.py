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


def _source_seen_before_time(test_csv: Path) -> np.ndarray:
    test = pd.read_csv(test_csv, usecols=["src", "time"])
    test = test.assign(_row_id=test.index.to_numpy(np.int64)).sort_values(["time", "_row_id"]).reset_index(drop=True)
    seen_before = np.zeros(len(test), dtype=np.int32)
    seen: Counter[int] = Counter()
    for _, group in test.groupby("time", sort=False):
        srcs = group["src"].to_numpy(np.int64)
        row_ids = group["_row_id"].to_numpy(np.int64)
        for src, rid in zip(srcs, row_ids):
            seen_before[int(rid)] = int(seen[int(src)])
        seen.update(int(x) for x in srcs)
    return seen_before


def _transplant_row(
    base: np.ndarray,
    expert: np.ndarray,
    *,
    mode: str,
    topn: int,
    min_overlap: int,
    blend_alpha: float,
) -> tuple[np.ndarray, bool, bool]:
    base_order = np.argsort(-base)
    expert_order = np.argsort(-expert)
    overlap = len(set(int(x) for x in base_order[:topn]).intersection(int(x) for x in expert_order[:topn]))
    agreed = overlap >= min_overlap
    if not agreed:
        return base, False, False

    sorted_scores = np.sort(base)[::-1]
    if mode == "full":
        new_order = expert_order
    elif mode == "topk":
        k = int(topn)
        used = set(int(x) for x in expert_order[:k])
        tail = [int(x) for x in base_order if int(x) not in used]
        new_order = np.asarray([int(x) for x in expert_order[:k]] + tail, dtype=np.int64)
    elif mode == "blend":
        new_order = np.argsort(-((1.0 - blend_alpha) * base + blend_alpha * expert))
    else:
        raise ValueError(mode)

    out = np.empty_like(base)
    out[new_order] = sorted_scores
    return out, not np.array_equal(new_order, base_order), True


def _write_dataset2(
    base_csv: Path,
    expert_csv: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    warm_seen: int,
    hot_seen: int,
    warm_mode: str,
    hot_mode: str,
    topn: int,
    min_overlap: int,
    blend_alpha: float,
) -> tuple[int, dict[str, float]]:
    seen = _source_seen_before_time(test_csv)
    rows = 0
    changed_rows = 0
    agree_rows = 0
    warm_rows = 0
    hot_rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
        expert_iter = pd.read_csv(expert_csv, header=None, chunksize=chunksize)
        for base_chunk, expert_chunk in zip(base_iter, expert_iter):
            base_mat = base_chunk.to_numpy(np.float64)
            expert_mat = expert_chunk.to_numpy(np.float64)
            out = np.empty_like(base_mat)
            for i in range(len(base_mat)):
                s = int(seen[offset + i])
                if s >= hot_seen:
                    mode = hot_mode
                    hot_rows += 1
                elif s >= warm_seen:
                    mode = warm_mode
                    warm_rows += 1
                else:
                    out[i] = base_mat[i]
                    continue
                row, changed, agreed = _transplant_row(
                    base_mat[i],
                    expert_mat[i],
                    mode=mode,
                    topn=topn,
                    min_overlap=min_overlap,
                    blend_alpha=blend_alpha,
                )
                out[i] = row
                changed_rows += int(changed)
                agree_rows += int(agreed)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in np.clip(row, 0.0, 1.0)])
            rows += len(base_mat)
            offset += len(base_mat)
    return rows, {
        "changed_rows": float(changed_rows),
        "agree_rows": float(agree_rows),
        "warm_rows": float(warm_rows),
        "hot_rows": float(hot_rows),
        "warm_seen": float(warm_seen),
        "hot_seen": float(hot_seen),
        "topn": float(topn),
        "min_overlap": float(min_overlap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warm-seen", type=int, default=4)
    parser.add_argument("--hot-seen", type=int, default=24)
    parser.add_argument("--warm-mode", choices=["full", "topk", "blend"], default="topk")
    parser.add_argument("--hot-mode", choices=["full", "topk", "blend"], default="full")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--min-overlap", type=int, default=1)
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--chunksize", type=int, default=8192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _write_dataset2(
        base_dir / "dataset2.csv",
        Path(args.expert_dir) / "dataset2.csv",
        Path(args.data_root) / "dataset2" / "test.csv",
        output_dir / "dataset2.csv",
        args.chunksize,
        args.warm_seen,
        args.hot_seen,
        args.warm_mode,
        args.hot_mode,
        args.topn,
        args.min_overlap,
        args.blend_alpha,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "stats": stats})


if __name__ == "__main__":
    main()
