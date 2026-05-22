from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sequential_heuristic import SEQUENTIAL_DEFAULT_WEIGHTS, build_sequential_stats, score_sequential_candidates
from simple_heuristic import history_only
from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.train_utils import load_json


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, len(values) - 1)


def _prob(values: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    values = values.astype(np.float64)
    if mode == "rank":
        return _rank01(values)
    if mode == "row_sigmoid":
        z = (values - float(values.mean())) / max(float(values.std()) * temperature, 1e-6)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    if mode == "minmax":
        lo = float(values.min())
        hi = float(values.max())
        return (values - lo) / max(hi - lo, 1e-12)
    raise ValueError(mode)


def _candidate_pool(test_path: Path, cand_cols: list[str]) -> set[int]:
    out: set[int] = set()
    for chunk in pd.read_csv(test_path, usecols=cand_cols, chunksize=8192):
        out.update(int(x) for x in chunk.to_numpy(np.int64).reshape(-1))
    return out


def _write_dataset(
    dataset: str,
    data_root: Path,
    model_dir: Path,
    output_csv: Path,
    include_valid_history: bool,
    chunk_size: int,
    mode: str,
    temperature: float,
    max_history_per_src: int,
    max_pairs_per_src: int,
) -> int:
    data_dir = data_root / dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=include_valid_history)
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    pool = _candidate_pool(test_path, cand_cols)
    stats = build_sequential_stats(
        train,
        candidate_pool=pool,
        max_history_per_src=max_history_per_src,
        max_pairs_per_src=max_pairs_per_src,
    )
    model_path = model_dir / f"{dataset}_sequential_heuristic.json"
    weights = np.asarray(load_json(model_path)["weights"], dtype=np.float64) if model_path.exists() else SEQUENTIAL_DEFAULT_WEIGHTS
    print({"dataset": dataset, "history_edges": len(train), "candidate_pool": len(pool), "transitions": len(stats.trans_count)}, flush=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            cmat = chunk[cand_cols].to_numpy(np.int64)
            srcs = chunk["src"].to_numpy(np.int64)
            times = chunk["time"].to_numpy(np.float64) if "time" in chunk.columns else np.zeros(len(chunk), dtype=np.float64)
            for i in range(len(chunk)):
                raw = score_sequential_candidates(int(srcs[i]), float(times[i]), cmat[i], stats, weights)
                prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/sequential_heuristic_hard")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_sequential_hard_rank")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "row_sigmoid", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--max-history-per-src", type=int, default=30)
    parser.add_argument("--max-pairs-per-src", type=int, default=30)
    args = parser.parse_args()
    rows = {}
    output_dir = Path(args.output_dir)
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
            max_history_per_src=args.max_history_per_src,
            max_pairs_per_src=args.max_pairs_per_src,
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode})


if __name__ == "__main__":
    main()
