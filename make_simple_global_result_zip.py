from __future__ import annotations

import argparse
import csv
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from simple_heuristic import DEFAULT_WEIGHTS, build_simple_stats, history_only, score_candidates
from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.train_utils import load_json


def _stats_for_dataset(dataset: str, data_root: Path, model_dir: Path, include_valid_history: bool, chunk_size: int):
    data_dir = data_root / dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=include_valid_history)
    stats = build_simple_stats(train)
    model_path = model_dir / f"{dataset}_simple_heuristic.json"
    weights = np.asarray(load_json(model_path)["weights"], dtype=np.float64) if model_path.exists() else DEFAULT_WEIGHTS
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    n_total = 0
    sum_x = 0.0
    sum_x2 = 0.0
    min_x = float("inf")
    max_x = float("-inf")
    for chunk in pd.read_csv(test_path, chunksize=chunk_size):
        cmat = chunk[cand_cols].to_numpy(np.int64)
        srcs = chunk["src"].to_numpy(np.int64)
        times = chunk["time"].to_numpy(np.float64) if "time" in chunk.columns else np.zeros(len(chunk), dtype=np.float64)
        for i in range(len(chunk)):
            raw = score_candidates(int(srcs[i]), float(times[i]), cmat[i], stats, weights)
            n_total += len(raw)
            sum_x += float(raw.sum())
            sum_x2 += float((raw * raw).sum())
            min_x = min(min_x, float(raw.min()))
            max_x = max(max_x, float(raw.max()))
    mean = sum_x / max(1, n_total)
    var = max(1e-12, sum_x2 / max(1, n_total) - mean * mean)
    return stats, weights, {"mean": mean, "std": math.sqrt(var), "min": min_x, "max": max_x, "count": n_total}


def _calibrate(raw: np.ndarray, params: dict, mode: str, temperature: float) -> np.ndarray:
    if mode == "global_sigmoid":
        z = (raw - params["mean"]) / max(params["std"] * temperature, 1e-6)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    if mode == "global_minmax":
        return (raw - params["min"]) / max(params["max"] - params["min"], 1e-12)
    if mode == "global_rankish":
        z = (raw - params["mean"]) / max(params["std"] * temperature, 1e-6)
        return np.clip(0.5 + z / 6.0, 0.0, 1.0)
    raise ValueError(mode)


def _write_dataset(dataset: str, data_root: Path, model_dir: Path, output_csv: Path, include_valid_history: bool, chunk_size: int, mode: str, temperature: float) -> int:
    stats, weights, params = _stats_for_dataset(dataset, data_root, model_dir, include_valid_history, chunk_size)
    print({"dataset": dataset, "calibration": params}, flush=True)
    data_dir = data_root / dataset
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            cmat = chunk[cand_cols].to_numpy(np.int64)
            srcs = chunk["src"].to_numpy(np.int64)
            times = chunk["time"].to_numpy(np.float64) if "time" in chunk.columns else np.zeros(len(chunk), dtype=np.float64)
            for i in range(len(chunk)):
                raw = score_candidates(int(srcs[i]), float(times[i]), cmat[i], stats, weights)
                prob = np.clip(_calibrate(raw, params, mode, temperature), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/simple_heuristic_hard_shuffled")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_global_sigmoid")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="global_sigmoid", choices=["global_sigmoid", "global_minmax", "global_rankish"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset] = _write_dataset(
            dataset,
            Path(args.data_root),
            Path(args.model_dir),
            output_dir / f"{dataset}.csv",
            include_valid_history=args.include_valid_history,
            chunk_size=args.chunk_size,
            mode=args.mode,
            temperature=args.temperature,
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode, "temperature": args.temperature})


if __name__ == "__main__":
    main()
