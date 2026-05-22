from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROWS = {"dataset1": 61051, "dataset2": 153420}


def _candidate_cols(columns) -> list[str]:
    cols = [c for c in columns if str(c).lower().startswith("c") and str(c)[1:].isdigit()]
    return sorted(cols, key=lambda x: int(str(x)[1:]))


def write_dataset(data_root: Path, dataset: str, output_csv: Path, mode: str) -> int:
    test_path = data_root / dataset / "test.csv"
    cols = _candidate_cols(pd.read_csv(test_path, nrows=0).columns)
    if len(cols) != 100:
        raise ValueError(f"{test_path} expected 100 candidate columns, got {len(cols)}")
    if mode == "linear":
        probs = np.linspace(1.0, 0.0, len(cols), dtype=np.float64)
    elif mode == "log":
        raw = 1.0 / np.log2(np.arange(2, len(cols) + 2, dtype=np.float64))
        probs = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-12)
    elif mode == "soft":
        raw = -np.arange(len(cols), dtype=np.float64) / 20.0
        exp = np.exp(raw - raw.max())
        probs = exp / exp.max()
    else:
        raise ValueError(mode)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, usecols=["src"], chunksize=8192):
            for _ in range(len(chunk)):
                writer.writerow([f"{float(x):.8f}" for x in probs])
            rows += len(chunk)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_order_prior")
    parser.add_argument("--mode", type=str, default="linear", choices=["linear", "log", "soft"])
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset] = write_dataset(Path(args.data_root), dataset, output_dir / f"{dataset}.csv", args.mode)
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode})


if __name__ == "__main__":
    main()
