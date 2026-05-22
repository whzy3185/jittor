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


def _rank_positions(order: np.ndarray) -> np.ndarray:
    pos = np.empty_like(order)
    pos[order] = np.arange(len(order), dtype=np.int64)
    return pos


def _intersection_row(
    base: np.ndarray,
    expert_a: np.ndarray,
    expert_b: np.ndarray,
    *,
    topn: int,
    take: int,
    base_keep: int,
    require_base_overlap: int,
) -> tuple[np.ndarray, bool, int]:
    base_order = np.argsort(-base)
    a_order = np.argsort(-expert_a)
    b_order = np.argsort(-expert_b)
    apos = _rank_positions(a_order)
    bpos = _rank_positions(b_order)
    base_top = set(int(x) for x in base_order[:base_keep])
    shared = [int(x) for x in a_order[:topn] if int(x) in set(int(y) for y in b_order[:topn])]
    if not shared:
        return base, False, 0
    shared.sort(key=lambda x: int(apos[x]) + int(bpos[x]))
    selected = shared[:take]
    overlap = len(set(selected).intersection(base_top))
    if overlap < require_base_overlap:
        return base, False, len(shared)

    used = set(selected)
    tail = [int(x) for x in base_order if int(x) not in used]
    new_order = np.asarray(selected + tail, dtype=np.int64)
    out = np.empty_like(base)
    out[new_order] = np.sort(base)[::-1]
    return out, not np.array_equal(new_order, base_order), len(shared)


def _write_dataset2(
    base_csv: Path,
    expert_a_csv: Path,
    expert_b_csv: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    warm_seen: int,
    hot_seen: int,
    warm_topn: int,
    hot_topn: int,
    warm_take: int,
    hot_take: int,
    base_keep: int,
    require_base_overlap: int,
) -> tuple[int, dict[str, float]]:
    seen = _source_seen_before_time(test_csv)
    rows = 0
    changed_rows = 0
    warm_rows = 0
    hot_rows = 0
    total_shared = 0
    eligible_rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
        a_iter = pd.read_csv(expert_a_csv, header=None, chunksize=chunksize)
        b_iter = pd.read_csv(expert_b_csv, header=None, chunksize=chunksize)
        for base_chunk, a_chunk, b_chunk in zip(base_iter, a_iter, b_iter):
            base_mat = base_chunk.to_numpy(np.float64)
            a_mat = a_chunk.to_numpy(np.float64)
            b_mat = b_chunk.to_numpy(np.float64)
            out = np.empty_like(base_mat)
            for i in range(len(base_mat)):
                s = int(seen[offset + i])
                if s >= hot_seen:
                    topn = hot_topn
                    take = hot_take
                    hot_rows += 1
                elif s >= warm_seen:
                    topn = warm_topn
                    take = warm_take
                    warm_rows += 1
                else:
                    out[i] = base_mat[i]
                    continue
                row, changed, shared = _intersection_row(
                    base_mat[i],
                    a_mat[i],
                    b_mat[i],
                    topn=topn,
                    take=take,
                    base_keep=base_keep,
                    require_base_overlap=require_base_overlap,
                )
                out[i] = row
                changed_rows += int(changed)
                total_shared += int(shared)
                eligible_rows += 1
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in np.clip(row, 0.0, 1.0)])
            rows += len(base_mat)
            offset += len(base_mat)
    return rows, {
        "changed_rows": float(changed_rows),
        "eligible_rows": float(eligible_rows),
        "warm_rows": float(warm_rows),
        "hot_rows": float(hot_rows),
        "avg_shared": float(total_shared / max(1, eligible_rows)),
        "warm_seen": float(warm_seen),
        "hot_seen": float(hot_seen),
        "warm_topn": float(warm_topn),
        "hot_topn": float(hot_topn),
        "warm_take": float(warm_take),
        "hot_take": float(hot_take),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-a-dir", required=True)
    parser.add_argument("--expert-b-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warm-seen", type=int, default=4)
    parser.add_argument("--hot-seen", type=int, default=24)
    parser.add_argument("--warm-topn", type=int, default=30)
    parser.add_argument("--hot-topn", type=int, default=50)
    parser.add_argument("--warm-take", type=int, default=8)
    parser.add_argument("--hot-take", type=int, default=20)
    parser.add_argument("--base-keep", type=int, default=30)
    parser.add_argument("--require-base-overlap", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=8192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _write_dataset2(
        base_dir / "dataset2.csv",
        Path(args.expert_a_dir) / "dataset2.csv",
        Path(args.expert_b_dir) / "dataset2.csv",
        Path(args.data_root) / "dataset2" / "test.csv",
        output_dir / "dataset2.csv",
        args.chunksize,
        args.warm_seen,
        args.hot_seen,
        args.warm_topn,
        args.hot_topn,
        args.warm_take,
        args.hot_take,
        args.base_keep,
        args.require_base_overlap,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "stats": stats})


if __name__ == "__main__":
    main()
