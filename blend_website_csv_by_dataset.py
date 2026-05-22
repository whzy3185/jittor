from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_weights(text: str, n: int) -> np.ndarray:
    values = np.asarray([float(x) for x in text.split(",") if x.strip() != ""], dtype=np.float64)
    if len(values) != n:
        raise ValueError(f"Expected {n} comma-separated weights, got {len(values)} from {text!r}")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return values / total


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
    parser.add_argument("--weights-dataset1", required=True, help="Comma-separated weights matching --inputs.")
    parser.add_argument("--weights-dataset2", required=True, help="Comma-separated weights matching --inputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunksize", type=int, default=8192)
    parser.add_argument("--shrink-dataset1", type=float, default=0.0)
    parser.add_argument("--shrink-dataset2", type=float, default=0.0)
    args = parser.parse_args()
    inputs = [Path(x) for x in args.inputs]
    w1 = _parse_weights(args.weights_dataset1, len(inputs))
    w2 = _parse_weights(args.weights_dataset2, len(inputs))
    output_dir = Path(args.output_dir)
    rows = {}
    rows["dataset1"] = blend_file(
        [p / "dataset1.csv" for p in inputs],
        w1,
        output_dir / "dataset1.csv",
        args.chunksize,
        min(max(float(args.shrink_dataset1), 0.0), 1.0),
    )
    rows["dataset2"] = blend_file(
        [p / "dataset2.csv" for p in inputs],
        w2,
        output_dir / "dataset2.csv",
        args.chunksize,
        min(max(float(args.shrink_dataset2), 0.0), 1.0),
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "weights_dataset1": w1.tolist(), "weights_dataset2": w2.tolist()})


if __name__ == "__main__":
    main()
