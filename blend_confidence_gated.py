from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _confidence(row: np.ndarray) -> float:
    if row.size < 2:
        return 0.0
    top = np.partition(row, -2)[-2:]
    return float(top[1] - top[0])


def _blend_dataset(
    base_csv: Path,
    expert_csv: Path,
    output_csv: Path,
    *,
    low_weight: float,
    mid_weight: float,
    high_weight: float,
    q_low: float,
    q_high: float,
    chunksize: int,
) -> tuple[int, dict[str, float]]:
    # First pass: estimate per-row confidence thresholds from the expert output.
    confs: list[np.ndarray] = []
    for chunk in pd.read_csv(expert_csv, header=None, chunksize=chunksize):
        arr = chunk.to_numpy(np.float64)
        confs.append(np.asarray([_confidence(row) for row in arr], dtype=np.float64))
    all_conf = np.concatenate(confs)
    t_low = float(np.quantile(all_conf, q_low))
    t_high = float(np.quantile(all_conf, q_high))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    low_rows = 0
    mid_rows = 0
    high_rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
        expert_iter = pd.read_csv(expert_csv, header=None, chunksize=chunksize)
        for base_chunk, expert_chunk in zip(base_iter, expert_iter):
            base = base_chunk.to_numpy(np.float64)
            expert = expert_chunk.to_numpy(np.float64)
            conf = np.asarray([_confidence(row) for row in expert], dtype=np.float64)
            weights = np.full(len(conf), mid_weight, dtype=np.float64)
            low_mask = conf < t_low
            high_mask = conf >= t_high
            weights[low_mask] = low_weight
            weights[high_mask] = high_weight
            low_rows += int(low_mask.sum())
            high_rows += int(high_mask.sum())
            mid_rows += int(len(conf) - low_mask.sum() - high_mask.sum())
            out = (1.0 - weights[:, None]) * base + weights[:, None] * expert
            out = np.clip(out, 0.0, 1.0)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(out)
    return rows, {
        "threshold_low": t_low,
        "threshold_high": t_high,
        "low_rows": float(low_rows),
        "mid_rows": float(mid_rows),
        "high_rows": float(high_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-weight", type=float, default=0.15)
    parser.add_argument("--mid-weight", type=float, default=0.35)
    parser.add_argument("--high-weight", type=float, default=0.55)
    parser.add_argument("--q-low", type=float, default=0.35)
    parser.add_argument("--q-high", type=float, default=0.85)
    parser.add_argument("--chunksize", type=int, default=8192)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    expert_dir = Path(args.expert_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset1_rows = 0
    with (output_dir / "dataset1.csv").open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunk in pd.read_csv(base_dir / "dataset1.csv", header=None, chunksize=args.chunksize):
            for row in chunk.to_numpy(np.float64):
                writer.writerow([f"{float(x):.8f}" for x in row])
            dataset1_rows += len(chunk)

    dataset2_rows, stats = _blend_dataset(
        base_dir / "dataset2.csv",
        expert_dir / "dataset2.csv",
        output_dir / "dataset2.csv",
        low_weight=args.low_weight,
        mid_weight=args.mid_weight,
        high_weight=args.high_weight,
        q_low=args.q_low,
        q_high=args.q_high,
        chunksize=args.chunksize,
    )

    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print(
        {
            "zip": str(zip_path),
            "rows": {"dataset1": dataset1_rows, "dataset2": dataset2_rows},
            "weights": {
                "low": args.low_weight,
                "mid": args.mid_weight,
                "high": args.high_weight,
            },
            "quantiles": {"low": args.q_low, "high": args.q_high},
            "stats": stats,
        }
    )


if __name__ == "__main__":
    main()
