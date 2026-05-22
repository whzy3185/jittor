from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _overlap_count(a: np.ndarray, b: np.ndarray, k: int) -> int:
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    return len(ia & ib)


def _row_weight(base: np.ndarray, expert: np.ndarray, low: float, mid: float, high: float) -> float:
    top1_same = int(np.argmax(base)) == int(np.argmax(expert))
    top5_overlap = _overlap_count(base, expert, 5)
    top10_overlap = _overlap_count(base, expert, 10)
    if top1_same or top5_overlap >= 3:
        return high
    if top10_overlap >= 4:
        return mid
    return low


def _copy_dataset1(base_dir: Path, output_dir: Path, chunksize: int) -> int:
    rows = 0
    with (output_dir / "dataset1.csv").open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunk in pd.read_csv(base_dir / "dataset1.csv", header=None, chunksize=chunksize):
            for row in chunk.to_numpy(np.float64):
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(chunk)
    return rows


def _blend_dataset2(
    base_dir: Path,
    expert_dir: Path,
    output_dir: Path,
    low_weight: float,
    mid_weight: float,
    high_weight: float,
    chunksize: int,
) -> tuple[int, dict[str, int]]:
    rows = 0
    low_rows = 0
    mid_rows = 0
    high_rows = 0
    with (output_dir / "dataset2.csv").open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        base_iter = pd.read_csv(base_dir / "dataset2.csv", header=None, chunksize=chunksize)
        expert_iter = pd.read_csv(expert_dir / "dataset2.csv", header=None, chunksize=chunksize)
        for base_chunk, expert_chunk in zip(base_iter, expert_iter):
            base = base_chunk.to_numpy(np.float64)
            expert = expert_chunk.to_numpy(np.float64)
            weights = np.zeros(len(base), dtype=np.float64)
            for i, (brow, erow) in enumerate(zip(base, expert)):
                w = _row_weight(brow, erow, low_weight, mid_weight, high_weight)
                weights[i] = w
                if w == high_weight:
                    high_rows += 1
                elif w == mid_weight:
                    mid_rows += 1
                else:
                    low_rows += 1
            out = (1.0 - weights[:, None]) * base + weights[:, None] * expert
            out = np.clip(out, 0.0, 1.0)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base)
    return rows, {"low_rows": low_rows, "mid_rows": mid_rows, "high_rows": high_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-weight", type=float, default=0.12)
    parser.add_argument("--mid-weight", type=float, default=0.35)
    parser.add_argument("--high-weight", type=float, default=0.58)
    parser.add_argument("--chunksize", type=int, default=8192)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    expert_dir = Path(args.expert_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _blend_dataset2(
        base_dir,
        expert_dir,
        output_dir,
        args.low_weight,
        args.mid_weight,
        args.high_weight,
        args.chunksize,
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
            "stats": stats,
        }
    )


if __name__ == "__main__":
    main()
