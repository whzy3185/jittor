from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.features import build_feature_builder, score_candidate_features_from_builder
from track1_dynamic_rec.train_utils import load_json


def _find_col(columns, aliases):
    lower = {str(c).lower(): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def _rank01_matrix(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, axis=1), axis=1)
    return order.astype(np.float32) / max(1, values.shape[1] - 1)


def _history_frame(train_path: Path, dataset: str, include_valid_history: bool) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    if dataset == "dataset2" and "split" in train.columns and not include_valid_history:
        split = train["split"]
        if split.dtype.kind in {"i", "u", "f"}:
            train = train[split.astype(int) == 0].copy()
        else:
            norm = split.astype(str).str.lower().str.strip()
            train = train[norm.isin(["0", "train", "training"])].copy()
    return train[["src", "dst", "time"]].copy()


def write_dataset_csv(
    dataset: str,
    data_root: Path,
    model_dir: Path,
    output_csv: Path,
    chunk_size: int,
    include_valid_history: bool,
) -> int:
    model = load_json(model_dir / f"{dataset}_linear_heuristic.json")
    data_dir = data_root / dataset
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    header = pd.read_csv(test_path, nrows=0)
    candidate_cols = wide_candidate_columns(header.columns)
    src_col = _find_col(header.columns, ("src", "source", "user", "user_id", "u"))
    time_col = _find_col(header.columns, ("time", "timestamp", "ts", "t"))
    if src_col is None or time_col is None or not candidate_cols:
        raise ValueError(f"{test_path} is not official wide test format")

    history = _history_frame(train_path, dataset, include_valid_history=include_valid_history)
    builder = build_feature_builder(history, window=float(model["window"]))
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    chunk_id = 0
    usecols = [src_col, time_col] + candidate_cols
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, usecols=usecols, chunksize=chunk_size):
            chunk_id += 1
            n = len(chunk)
            k = len(candidate_cols)
            src = pd.to_numeric(chunk[src_col], errors="raise").to_numpy(np.int64)
            time = pd.to_numeric(chunk[time_col], errors="coerce").fillna(0).to_numpy(np.float32)
            cands = chunk[candidate_cols].apply(pd.to_numeric, errors="raise").to_numpy(np.int64)
            flat = pd.DataFrame(
                {
                    "src": np.repeat(src, k),
                    "dst": cands.reshape(-1),
                    "time": np.repeat(time, k),
                }
            )
            feat, heur = score_candidate_features_from_builder(flat, builder)
            X = np.column_stack([feat, heur.astype(np.float32)])
            score = ((X - mean) / std) @ weights
            probs = _rank01_matrix(score.reshape(n, k))
            for row in probs:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += n
            if chunk_id == 1 or chunk_id % 10 == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/heuristic_linear_tuned")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_linear")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--datasets", nargs="*", default=["dataset1", "dataset2"], choices=["dataset1", "dataset2"])
    parser.add_argument("--include-valid-history", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for dataset in args.datasets:
        rows[dataset] = write_dataset_csv(
            dataset,
            data_root,
            model_dir,
            output_dir / f"{dataset}.csv",
            chunk_size=args.chunk_size,
            include_valid_history=args.include_valid_history,
        )
    zip_path = output_dir / "result.zip"
    if (output_dir / "dataset1.csv").exists() and (output_dir / "dataset2.csv").exists():
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
            zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
        print({"zip": str(zip_path), "rows": rows})
    else:
        print({"zip": None, "rows": rows, "note": "both dataset1.csv and dataset2.csv are required to build result.zip"})


if __name__ == "__main__":
    main()
