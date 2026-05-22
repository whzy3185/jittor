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


def _causal_source_weights(
    test_csv: Path,
    cold_weight: float,
    warm_weight: float,
    hot_weight: float,
    warm_seen: int,
    hot_seen: int,
) -> tuple[np.ndarray, dict[str, float]]:
    test = pd.read_csv(test_csv, usecols=["src", "time"])
    test = test.assign(_row_id=test.index.to_numpy(np.int64)).sort_values(["time", "_row_id"]).reset_index(drop=True)
    weights = np.zeros(len(test), dtype=np.float64)
    seen: Counter[int] = Counter()
    cold = warm = hot = 0
    groups = 0
    for _, group in test.groupby("time", sort=False):
        srcs = group["src"].to_numpy(np.int64)
        row_ids = group["_row_id"].to_numpy(np.int64)
        for src_raw, row_id_raw in zip(srcs, row_ids):
            cnt = int(seen[int(src_raw)])
            if cnt >= hot_seen:
                w = hot_weight
                hot += 1
            elif cnt >= warm_seen:
                w = warm_weight
                warm += 1
            else:
                w = cold_weight
                cold += 1
            weights[int(row_id_raw)] = w
        seen.update(int(x) for x in srcs)
        groups += 1
    return weights, {
        "time_groups": float(groups),
        "cold_rows": float(cold),
        "warm_rows": float(warm),
        "hot_rows": float(hot),
        "warm_seen": float(warm_seen),
        "hot_seen": float(hot_seen),
    }


def _blend_dataset2(
    base_csv: Path,
    expert_csv: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    cold_weight: float,
    warm_weight: float,
    hot_weight: float,
    warm_seen: int,
    hot_seen: int,
) -> tuple[int, dict[str, float]]:
    weights, stats = _causal_source_weights(test_csv, cold_weight, warm_weight, hot_weight, warm_seen, hot_seen)
    rows = 0
    offset = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
        expert_iter = pd.read_csv(expert_csv, header=None, chunksize=chunksize)
        for base_chunk, expert_chunk in zip(base_iter, expert_iter):
            base = base_chunk.to_numpy(np.float64)
            expert = expert_chunk.to_numpy(np.float64)
            w = weights[offset : offset + len(base)].reshape((-1, 1))
            out = np.clip((1.0 - w) * base + w * expert, 0.0, 1.0)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base)
            offset += len(base)
    stats["mean_weight"] = float(weights.mean())
    stats["max_weight"] = float(weights.max())
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cold-weight", type=float, default=0.01)
    parser.add_argument("--warm-weight", type=float, default=0.06)
    parser.add_argument("--hot-weight", type=float, default=0.14)
    parser.add_argument("--warm-seen", type=int, default=8)
    parser.add_argument("--hot-seen", type=int, default=40)
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
        args.cold_weight,
        args.warm_weight,
        args.hot_weight,
        args.warm_seen,
        args.hot_seen,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "stats": stats})


if __name__ == "__main__":
    main()
