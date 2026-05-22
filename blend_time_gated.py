from __future__ import annotations

import argparse
import csv
import zipfile
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


def _time_thresholds(test_csv: Path, q1: float, q2: float) -> tuple[float, float]:
    times = pd.read_csv(test_csv, usecols=["time"])["time"].to_numpy(np.float64)
    return float(np.quantile(times, q1)), float(np.quantile(times, q2))


def _write_dataset2(
    base_csv: Path,
    expert_csv: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    early_weight: float,
    mid_weight: float,
    late_weight: float,
    q1: float,
    q2: float,
) -> tuple[int, dict[str, float]]:
    t1, t2 = _time_thresholds(test_csv, q1, q2)
    rows = 0
    early_rows = 0
    mid_rows = 0
    late_rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
        expert_iter = pd.read_csv(expert_csv, header=None, chunksize=chunksize)
        test_iter = pd.read_csv(test_csv, usecols=["time"], chunksize=chunksize)
        for base_chunk, expert_chunk, time_chunk in zip(base_iter, expert_iter, test_iter):
            base = base_chunk.to_numpy(np.float64)
            expert = expert_chunk.to_numpy(np.float64)
            times = time_chunk["time"].to_numpy(np.float64)
            weights = np.empty(len(times), dtype=np.float64)
            early_mask = times < t1
            late_mask = times >= t2
            mid_mask = ~(early_mask | late_mask)
            weights[early_mask] = early_weight
            weights[mid_mask] = mid_weight
            weights[late_mask] = late_weight
            early_rows += int(early_mask.sum())
            mid_rows += int(mid_mask.sum())
            late_rows += int(late_mask.sum())
            out = (1.0 - weights[:, None]) * base + weights[:, None] * expert
            out = np.clip(out, 0.0, 1.0)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base)
    return rows, {
        "threshold_early": t1,
        "threshold_late": t2,
        "early_rows": float(early_rows),
        "mid_rows": float(mid_rows),
        "late_rows": float(late_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--early-weight", type=float, default=0.15)
    parser.add_argument("--mid-weight", type=float, default=0.35)
    parser.add_argument("--late-weight", type=float, default=0.62)
    parser.add_argument("--q1", type=float, default=0.30)
    parser.add_argument("--q2", type=float, default=0.75)
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
        args.early_weight,
        args.mid_weight,
        args.late_weight,
        args.q1,
        args.q2,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print(
        {
            "zip": str(zip_path),
            "rows": {"dataset1": rows1, "dataset2": rows2},
            "weights": {"early": args.early_weight, "mid": args.mid_weight, "late": args.late_weight},
            "quantiles": {"q1": args.q1, "q2": args.q2},
            "stats": stats,
        }
    )


if __name__ == "__main__":
    main()
