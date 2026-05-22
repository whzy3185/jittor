from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def blend_file(paths: list[Path], weights: np.ndarray, output_csv: Path, chunksize: int, shrink: float) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    iters = [pd.read_csv(p, header=None, chunksize=chunksize) for p in paths]
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunks in zip(*iters):
            mats = [c.to_numpy(np.float64) for c in chunks]
            out = np.zeros_like(mats[0], dtype=np.float64)
            for w, mat in zip(weights, mats):
                out += float(w) * mat
            if shrink > 0:
                out = (1.0 - shrink) * out + shrink * 0.5
            out = np.clip(out, 0.0, 1.0)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(out)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Directories containing dataset1.csv and dataset2.csv.")
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunksize", type=int, default=8192)
    parser.add_argument("--shrink", type=float, default=0.0, help="Move probabilities away from 0/1: p=(1-s)*p+s*0.5")
    args = parser.parse_args()
    if len(args.inputs) != len(args.weights):
        raise ValueError("--inputs and --weights lengths must match")
    weights = np.asarray(args.weights, dtype=np.float64)
    weights = weights / weights.sum()
    output_dir = Path(args.output_dir)
    shrink = min(max(float(args.shrink), 0.0), 1.0)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        paths = [Path(d) / f"{dataset}.csv" for d in args.inputs]
        rows[dataset] = blend_file(paths, weights, output_dir / f"{dataset}.csv", args.chunksize, shrink)
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "weights": weights.tolist(), "shrink": shrink})


if __name__ == "__main__":
    main()
