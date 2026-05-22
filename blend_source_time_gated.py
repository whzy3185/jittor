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


def _thresholds(test_csv: Path, chunksize: int, src_q: float, time_q: float) -> tuple[Counter[int], float, float]:
    counts = _source_counts(test_csv, chunksize)
    src_threshold = float(np.quantile(np.asarray(list(counts.values()), dtype=np.float64), src_q))
    time_values = pd.read_csv(test_csv, usecols=["time"])["time"].to_numpy(np.float64)
    time_threshold = float(np.quantile(time_values, time_q))
    return counts, src_threshold, time_threshold


def _write_dataset2(
    base_csv: Path,
    expert_csv: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    low_weight: float,
    active_weight: float,
    late_weight: float,
    active_late_weight: float,
    src_q: float,
    time_q: float,
) -> tuple[int, dict[str, float]]:
    counts, src_threshold, time_threshold = _thresholds(test_csv, chunksize, src_q, time_q)
    rows = 0
    low_rows = 0
    active_rows = 0
    late_rows = 0
    active_late_rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
        expert_iter = pd.read_csv(expert_csv, header=None, chunksize=chunksize)
        test_iter = pd.read_csv(test_csv, usecols=["src", "time"], chunksize=chunksize)
        for base_chunk, expert_chunk, test_chunk in zip(base_iter, expert_iter, test_iter):
            base = base_chunk.to_numpy(np.float64)
            expert = expert_chunk.to_numpy(np.float64)
            srcs = test_chunk["src"].to_numpy(np.int64)
            times = test_chunk["time"].to_numpy(np.float64)
            weights = np.empty(len(srcs), dtype=np.float64)
            for i, (src, time_value) in enumerate(zip(srcs, times)):
                active = float(counts[int(src)]) >= src_threshold
                late = float(time_value) >= time_threshold
                if active and late:
                    weights[i] = active_late_weight
                    active_late_rows += 1
                elif active:
                    weights[i] = active_weight
                    active_rows += 1
                elif late:
                    weights[i] = late_weight
                    late_rows += 1
                else:
                    weights[i] = low_weight
                    low_rows += 1
            out = (1.0 - weights[:, None]) * base + weights[:, None] * expert
            out = np.clip(out, 0.0, 1.0)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base)
    return rows, {
        "src_threshold": src_threshold,
        "time_threshold": time_threshold,
        "low_rows": float(low_rows),
        "active_rows": float(active_rows),
        "late_rows": float(late_rows),
        "active_late_rows": float(active_late_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-weight", type=float, default=0.12)
    parser.add_argument("--active-weight", type=float, default=0.38)
    parser.add_argument("--late-weight", type=float, default=0.42)
    parser.add_argument("--active-late-weight", type=float, default=0.68)
    parser.add_argument("--src-q", type=float, default=0.80)
    parser.add_argument("--time-q", type=float, default=0.75)
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
        args.active_weight,
        args.late_weight,
        args.active_late_weight,
        args.src_q,
        args.time_q,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print(
        {
            "zip": str(zip_path),
            "rows": {"dataset1": rows1, "dataset2": rows2},
            "weights": {
                "low": args.low_weight,
                "active": args.active_weight,
                "late": args.late_weight,
                "active_late": args.active_late_weight,
            },
            "stats": stats,
        }
    )


if __name__ == "__main__":
    main()
