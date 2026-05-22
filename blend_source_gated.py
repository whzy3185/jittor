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


def _source_counts(test_csv: Path, chunksize: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for chunk in pd.read_csv(test_csv, usecols=["src"], chunksize=chunksize):
        counts.update(int(x) for x in chunk["src"].to_numpy(np.int64))
    return counts


def _write_dataset2(
    base_csv: Path,
    expert_csv: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    low_weight: float,
    mid_weight: float,
    high_weight: float,
    q_mid: float,
    q_high: float,
) -> tuple[int, dict[str, float]]:
    counts = _source_counts(test_csv, chunksize)
    values = np.asarray(list(counts.values()), dtype=np.float64)
    t_mid = float(np.quantile(values, q_mid))
    t_high = float(np.quantile(values, q_high))
    rows = 0
    low_rows = 0
    mid_rows = 0
    high_rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
        expert_iter = pd.read_csv(expert_csv, header=None, chunksize=chunksize)
        test_iter = pd.read_csv(test_csv, usecols=["src"], chunksize=chunksize)
        for base_chunk, expert_chunk, src_chunk in zip(base_iter, expert_iter, test_iter):
            base = base_chunk.to_numpy(np.float64)
            expert = expert_chunk.to_numpy(np.float64)
            srcs = src_chunk["src"].to_numpy(np.int64)
            weights = np.empty(len(srcs), dtype=np.float64)
            for i, src in enumerate(srcs):
                cnt = float(counts[int(src)])
                if cnt >= t_high:
                    weights[i] = high_weight
                    high_rows += 1
                elif cnt >= t_mid:
                    weights[i] = mid_weight
                    mid_rows += 1
                else:
                    weights[i] = low_weight
                    low_rows += 1
            out = (1.0 - weights[:, None]) * base + weights[:, None] * expert
            out = np.clip(out, 0.0, 1.0)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base)
    return rows, {
        "src_unique": float(len(counts)),
        "threshold_mid": t_mid,
        "threshold_high": t_high,
        "low_rows": float(low_rows),
        "mid_rows": float(mid_rows),
        "high_rows": float(high_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-weight", type=float, default=0.10)
    parser.add_argument("--mid-weight", type=float, default=0.35)
    parser.add_argument("--high-weight", type=float, default=0.60)
    parser.add_argument("--q-mid", type=float, default=0.50)
    parser.add_argument("--q-high", type=float, default=0.80)
    parser.add_argument("--chunksize", type=int, default=8192)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    expert_dir = Path(args.expert_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _write_dataset2(
        base_dir / "dataset2.csv",
        expert_dir / "dataset2.csv",
        Path(args.data_root) / "dataset2" / "test.csv",
        output_dir / "dataset2.csv",
        args.chunksize,
        args.low_weight,
        args.mid_weight,
        args.high_weight,
        args.q_mid,
        args.q_high,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print(
        {
            "zip": str(zip_path),
            "rows": {"dataset1": rows1, "dataset2": rows2},
            "weights": {"low": args.low_weight, "mid": args.mid_weight, "high": args.high_weight},
            "quantiles": {"mid": args.q_mid, "high": args.q_high},
            "stats": stats,
        }
    )


if __name__ == "__main__":
    main()
