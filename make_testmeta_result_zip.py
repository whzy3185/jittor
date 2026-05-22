from __future__ import annotations

import argparse
import csv
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from track1_dynamic_rec.data import wide_candidate_columns


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, len(values) - 1)


def _prob(values: np.ndarray, mode: str) -> np.ndarray:
    values = values.astype(np.float64)
    if mode == "rank":
        return _rank01(values)
    if mode == "minmax":
        lo = float(values.min())
        hi = float(values.max())
        return (values - lo) / max(hi - lo, 1e-12)
    if mode == "row_sigmoid":
        z = (values - float(values.mean())) / max(float(values.std()), 1e-6)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(mode)


def _collect_test_meta(test_path: Path, cand_cols: list[str], chunk_size: int):
    global_count: Counter[int] = Counter()
    src_dst_count: dict[int, Counter[int]] = defaultdict(Counter)
    src_dst_col_sum: dict[int, Counter[int]] = defaultdict(Counter)
    for chunk in pd.read_csv(test_path, chunksize=chunk_size):
        srcs = chunk["src"].to_numpy(np.int64)
        mat = chunk[cand_cols].to_numpy(np.int64)
        for row_src, row in zip(srcs, mat):
            src = int(row_src)
            bucket = src_dst_count[src]
            col_bucket = src_dst_col_sum[src]
            for j, dst in enumerate(row):
                d = int(dst)
                global_count[d] += 1
                bucket[d] += 1
                col_bucket[d] += j + 1
    return global_count, src_dst_count, src_dst_col_sum


def _write_dataset(
    dataset: str,
    data_root: Path,
    output_csv: Path,
    chunk_size: int,
    mode: str,
    w_srcfreq: float,
    w_globalfreq: float,
    w_mean_col: float,
    w_col_prior: float,
) -> int:
    test_path = data_root / dataset / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    global_count, src_dst_count, src_dst_col_sum = _collect_test_meta(test_path, cand_cols, chunk_size)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            srcs = chunk["src"].to_numpy(np.int64)
            mat = chunk[cand_cols].to_numpy(np.int64)
            for src, row in zip(srcs, mat):
                src = int(src)
                raw = np.zeros(len(row), dtype=np.float64)
                sc = src_dst_count.get(src, {})
                scol = src_dst_col_sum.get(src, {})
                for j, dst in enumerate(row):
                    d = int(dst)
                    sf = int(sc.get(d, 0))
                    gf = int(global_count.get(d, 0))
                    mean_col = float(scol.get(d, j + 1)) / max(1, sf)
                    raw[j] = (
                        w_srcfreq * math.log1p(sf)
                        + w_globalfreq * math.log1p(gf)
                        + w_mean_col / math.log2(mean_col + 1.0)
                        + w_col_prior / math.log2(j + 2.0)
                    )
                prob = np.clip(_prob(raw, mode), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_testmeta_rank")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "minmax", "row_sigmoid"])
    parser.add_argument("--w-srcfreq", type=float, default=2.0)
    parser.add_argument("--w-globalfreq", type=float, default=0.35)
    parser.add_argument("--w-mean-col", type=float, default=0.8)
    parser.add_argument("--w-col-prior", type=float, default=0.2)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset] = _write_dataset(
            dataset,
            Path(args.data_root),
            output_dir / f"{dataset}.csv",
            args.chunk_size,
            args.mode,
            args.w_srcfreq,
            args.w_globalfreq,
            args.w_mean_col,
            args.w_col_prior,
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode})


if __name__ == "__main__":
    main()
