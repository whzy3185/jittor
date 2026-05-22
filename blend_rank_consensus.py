from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _rank01_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, values.size - 1)


def _consensus_row(rows: list[np.ndarray], sharpen: float) -> np.ndarray:
    ranks = np.stack([_rank01_desc(row) for row in rows], axis=0)
    mean_rank = ranks.mean(axis=0)
    if sharpen > 0:
        centered = mean_rank - mean_rank.mean()
        scaled = centered * sharpen
        scaled = np.clip(scaled, -30.0, 30.0)
        expv = np.exp(scaled)
        prob = expv / max(float(expv.max()), 1e-12)
        return np.clip(prob, 0.0, 1.0)
    return mean_rank


def _write_dataset2(
    base_csv: Path,
    expert_csvs: list[Path],
    output_csv: Path,
    weight: float,
    sharpen: float,
    chunksize: int,
) -> tuple[int, dict[str, float]]:
    rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base_iter = pd.read_csv(base_csv, header=None, chunksize=chunksize)
    expert_iters = [pd.read_csv(path, header=None, chunksize=chunksize) for path in expert_csvs]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunks in zip(base_iter, *expert_iters):
            base = chunks[0].to_numpy(np.float64)
            experts = [chunk.to_numpy(np.float64) for chunk in chunks[1:]]
            out = np.zeros_like(base)
            for i in range(len(base)):
                consensus = _consensus_row([expert[i] for expert in experts], sharpen)
                out[i] = (1.0 - weight) * base[i] + weight * consensus
            out = np.clip(out, 0.0, 1.0)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base)
    return rows, {"weight": float(weight), "sharpen": float(sharpen)}


def _copy_dataset1(base_dir: Path, output_dir: Path, chunksize: int) -> int:
    rows = 0
    with (output_dir / "dataset1.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(base_dir / "dataset1.csv", header=None, chunksize=chunksize):
            for row in chunk.to_numpy(np.float64):
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(chunk)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight", type=float, default=0.40)
    parser.add_argument("--sharpen", type=float, default=4.0)
    parser.add_argument("--chunksize", type=int, default=4096)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    expert_dirs = [Path(x) for x in args.expert_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _write_dataset2(
        base_dir / "dataset2.csv",
        [p / "dataset2.csv" for p in expert_dirs],
        output_dir / "dataset2.csv",
        args.weight,
        args.sharpen,
        args.chunksize,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "stats": stats})


if __name__ == "__main__":
    main()
