from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from simple_heuristic import build_simple_stats, component_matrix, history_only
from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.train_utils import load_json


PRIOR_IDX = 6


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, len(values) - 1)


def _prob(values: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    values = values.astype(np.float64)
    if mode == "rank":
        return _rank01(values)
    if mode == "minmax":
        lo = float(values.min())
        hi = float(values.max())
        return (values - lo) / max(hi - lo, 1e-12)
    if mode == "sigmoid":
        z = values / max(temperature, 1e-6)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    if mode == "row_sigmoid":
        z = (values - float(values.mean())) / max(float(values.std()), 1e-6)
        z = np.clip(z / max(temperature, 1e-6), -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Unknown mode: {mode}")


def _load_model(model_dir: Path, dataset: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]:
    payload = load_json(model_dir / f"{dataset}_simple_jittor.json")
    weights = np.asarray(payload["weights"], dtype=np.float64)
    bias = float(payload.get("bias", 0.0))
    mean = np.asarray(payload["mean"], dtype=np.float64)
    std = np.asarray(payload["std"], dtype=np.float64)
    std[std < 1e-6] = 1.0
    use_candidate_prior = bool(payload.get("use_candidate_prior", payload.get("args", {}).get("use_candidate_prior", False)))
    return weights, bias, mean, std, use_candidate_prior


def _write_dataset(
    dataset: str,
    data_root: Path,
    model_dir: Path,
    output_csv: Path,
    include_valid_history: bool,
    chunk_size: int,
    mode: str,
    temperature: float,
) -> int:
    data_dir = data_root / dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=include_valid_history)
    stats = build_simple_stats(train)
    weights, bias, mean, std, use_candidate_prior = _load_model(model_dir, dataset)
    test_path = data_dir / "test.csv"
    header = pd.read_csv(test_path, nrows=0)
    cand_cols = wide_candidate_columns(header.columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            cmat = chunk[cand_cols].to_numpy(np.int64)
            srcs = chunk["src"].to_numpy(np.int64)
            times = chunk["time"].to_numpy(np.float64) if "time" in chunk.columns else np.zeros(len(chunk), dtype=np.float64)
            for i in range(len(chunk)):
                feat = component_matrix(int(srcs[i]), float(times[i]), cmat[i], stats)
                if not use_candidate_prior:
                    feat[:, PRIOR_IDX] = 0.0
                raw = ((feat - mean) / std) @ weights + bias
                prob = _prob(raw, mode, temperature)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/simple_jittor")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_jittor_simple_rank")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "minmax", "sigmoid", "row_sigmoid"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset] = _write_dataset(
            dataset,
            data_root,
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
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode})


if __name__ == "__main__":
    main()
