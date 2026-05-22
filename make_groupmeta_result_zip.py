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


def _prob(values: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    values = values.astype(np.float64)
    if mode == "rank":
        return _rank01(values)
    if mode == "minmax":
        lo = float(values.min())
        hi = float(values.max())
        return (values - lo) / max(hi - lo, 1e-12)
    if mode == "row_sigmoid":
        z = (values - float(values.mean())) / max(float(values.std()) * temperature, 1e-6)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(mode)


class GroupMeta:
    def __init__(self):
        self.global_count: Counter[int] = Counter()
        self.time_count: dict[int, Counter[int]] = defaultdict(Counter)
        self.src_count: dict[int, Counter[int]] = defaultdict(Counter)
        self.src_time_count: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.src_col_sum: dict[int, Counter[int]] = defaultdict(Counter)
        self.src_time_col_sum: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.src_time_rows: Counter[tuple[int, int]] = Counter()


def _collect_meta(test_path: Path, cand_cols: list[str], chunk_size: int) -> GroupMeta:
    meta = GroupMeta()
    for chunk in pd.read_csv(test_path, chunksize=chunk_size):
        srcs = chunk["src"].to_numpy(np.int64)
        times = chunk["time"].to_numpy(np.int64)
        mat = chunk[cand_cols].to_numpy(np.int64)
        for src_raw, time_raw, row in zip(srcs, times, mat):
            src = int(src_raw)
            time_value = int(time_raw)
            key = (src, time_value)
            meta.src_time_rows[key] += 1
            st_count = meta.src_time_count[key]
            st_col = meta.src_time_col_sum[key]
            s_count = meta.src_count[src]
            s_col = meta.src_col_sum[src]
            t_count = meta.time_count[time_value]
            for j, dst_raw in enumerate(row):
                dst = int(dst_raw)
                col = j + 1
                meta.global_count[dst] += 1
                t_count[dst] += 1
                s_count[dst] += 1
                st_count[dst] += 1
                s_col[dst] += col
                st_col[dst] += col
    return meta


def _score_row(src: int, time_value: int, row: np.ndarray, meta: GroupMeta, weights: np.ndarray) -> np.ndarray:
    raw = np.zeros(len(row), dtype=np.float64)
    st_key = (int(src), int(time_value))
    st_count = meta.src_time_count.get(st_key, {})
    st_col = meta.src_time_col_sum.get(st_key, {})
    src_count = meta.src_count.get(int(src), {})
    src_col = meta.src_col_sum.get(int(src), {})
    time_count = meta.time_count.get(int(time_value), {})
    group_rows = max(1, int(meta.src_time_rows.get(st_key, 1)))
    for j, dst_raw in enumerate(row):
        dst = int(dst_raw)
        stf = int(st_count.get(dst, 0))
        sf = int(src_count.get(dst, 0))
        tf = int(time_count.get(dst, 0))
        gf = int(meta.global_count.get(dst, 0))
        st_mean_col = float(st_col.get(dst, j + 1)) / max(1, stf)
        src_mean_col = float(src_col.get(dst, j + 1)) / max(1, sf)
        st_share = stf / group_rows
        col_prior = 1.0 / math.log2(j + 2.0)
        features = np.asarray(
            [
                math.log1p(stf),
                st_share,
                1.0 / math.log2(st_mean_col + 1.0),
                math.log1p(sf),
                1.0 / math.log2(src_mean_col + 1.0),
                math.log1p(tf),
                math.log1p(gf),
                col_prior,
            ],
            dtype=np.float64,
        )
        raw[j] = float(features @ weights)
    return raw


def _write_dataset(
    dataset: str,
    data_root: Path,
    output_csv: Path,
    chunk_size: int,
    mode: str,
    temperature: float,
    weights: np.ndarray,
) -> int:
    test_path = data_root / dataset / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    meta = _collect_meta(test_path, cand_cols, chunk_size)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            srcs = chunk["src"].to_numpy(np.int64)
            times = chunk["time"].to_numpy(np.int64)
            mat = chunk[cand_cols].to_numpy(np.int64)
            for src, time_value, row in zip(srcs, times, mat):
                raw = _score_row(int(src), int(time_value), row, meta, weights)
                prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    print(
        {
            "dataset": dataset,
            "rows": rows,
            "global_candidates": len(meta.global_count),
            "src_time_groups": len(meta.src_time_rows),
        },
        flush=True,
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_groupmeta_rank")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "minmax", "row_sigmoid"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--weights",
        type=str,
        default="2.4,1.8,1.0,1.4,0.6,0.35,0.15,0.15",
        help="Comma-separated feature weights: st_log,st_share,st_col,src_log,src_col,time_log,global_log,col_prior.",
    )
    args = parser.parse_args()
    weights = np.asarray([float(x) for x in args.weights.split(",")], dtype=np.float64)
    if len(weights) != 8:
        raise ValueError("--weights must contain 8 numbers")
    output_dir = Path(args.output_dir)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset] = _write_dataset(
            dataset,
            Path(args.data_root),
            output_dir / f"{dataset}.csv",
            args.chunk_size,
            args.mode,
            args.temperature,
            weights,
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode, "weights": weights.tolist()})


if __name__ == "__main__":
    main()
