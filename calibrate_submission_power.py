from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _calibrate(values: np.ndarray, gamma: float, mix: float) -> np.ndarray:
    values = np.clip(values.astype(np.float64), 0.0, 1.0)
    powered = np.power(values, gamma)
    if mix > 0:
        powered = (1.0 - mix) * powered + mix * values
    return np.clip(powered, 0.0, 1.0)


def _write_file(input_csv: Path, output_csv: Path, gamma: float, mix: float, chunksize: int) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunk in pd.read_csv(input_csv, header=None, chunksize=chunksize):
            out = _calibrate(chunk.to_numpy(np.float64), gamma, mix)
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--mix-original", type=float, default=0.0)
    parser.add_argument("--chunksize", type=int, default=8192)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset] = _write_file(
            input_dir / f"{dataset}.csv",
            output_dir / f"{dataset}.csv",
            float(args.gamma),
            float(args.mix_original),
            int(args.chunksize),
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "gamma": float(args.gamma), "mix_original": float(args.mix_original)})


if __name__ == "__main__":
    main()
