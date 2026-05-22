from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _score_cols(columns) -> list[str]:
    cols = [c for c in columns if str(c).lower().startswith("c") and str(c)[1:].isdigit()]
    return sorted(cols, key=lambda x: int(str(x)[1:]))


def _row_probabilities(values: np.ndarray, mode: str) -> np.ndarray:
    values = values.astype(np.float64)
    if mode == "minmax":
        lo = values.min(axis=1, keepdims=True)
        hi = values.max(axis=1, keepdims=True)
        denom = np.maximum(hi - lo, 1e-12)
        return (values - lo) / denom
    if mode == "rank":
        order = np.argsort(np.argsort(values, axis=1), axis=1)
        return order.astype(np.float64) / max(1, values.shape[1] - 1)
    if mode == "sigmoid":
        mean = values.mean(axis=1, keepdims=True)
        std = np.maximum(values.std(axis=1, keepdims=True), 1e-6)
        z = np.clip((values - mean) / std, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Unknown mode: {mode}")


def convert_scores(scores_path: Path, output_csv: Path, mode: str, chunksize: int = 8192) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(scores_path, nrows=0)
    cols = _score_cols(header.columns)
    if not cols:
        raise ValueError(f"{scores_path} has no c1..cK columns")
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(scores_path, usecols=cols, chunksize=chunksize):
            values = chunk.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
            probs = np.clip(_row_probabilities(values, mode), 0.0, 1.0)
            for row in probs:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(chunk)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-root", type=str, default="outputs/track1_heuristic_aggressive")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission")
    parser.add_argument("--zip-name", type=str, default="result.zip")
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "minmax", "sigmoid"])
    parser.add_argument("--chunksize", type=int, default=8192)
    args = parser.parse_args()

    scores_root = Path(args.scores_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        scores_path = scores_root / dataset / "result_scores.csv"
        if not scores_path.exists():
            raise FileNotFoundError(scores_path)
        rows[dataset] = convert_scores(scores_path, output_dir / f"{dataset}.csv", args.mode, chunksize=args.chunksize)
    zip_path = output_dir / args.zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dataset in ("dataset1", "dataset2"):
            zf.write(output_dir / f"{dataset}.csv", arcname=f"{dataset}.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode})


if __name__ == "__main__":
    main()
