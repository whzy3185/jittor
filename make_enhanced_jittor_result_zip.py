from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt

from enhanced_heuristic import ENHANCED_COMPONENT_NAMES, build_enhanced_stats, enhanced_component_matrix
from simple_heuristic import history_only
from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.train_utils import load_json
from train_enhanced_jittor import EnhancedMLPRanker


PRIOR_IDX = ENHANCED_COMPONENT_NAMES.index("candidate_prior")


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


def _load_model(model_dir: Path, dataset: str):
    meta = load_json(model_dir / f"{dataset}_enhanced_mlp.json")
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    model = EnhancedMLPRanker(len(mean), hidden_dim=int(meta["hidden_dim"]), dropout=0.0)
    state = jt.load(str(model_dir / meta["checkpoint"]))
    model.load_state_dict(state)
    model.eval()
    return model, mean, std, bool(meta.get("use_candidate_prior", False)), meta


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
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=include_valid_history)
    pool = _candidate_pool(test_path, cand_cols)
    stats = build_enhanced_stats(train, candidate_pool=pool, max_history_per_src=max_history_per_src, max_pairs_per_src=max_pairs_per_src)
    model, mean, std, use_candidate_prior, _ = _load_model(model_dir, dataset)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            cmat = chunk[cand_cols].to_numpy(np.int64)
            srcs = chunk["src"].to_numpy(np.int64)
            times = chunk["time"].to_numpy(np.float64) if "time" in chunk.columns else np.zeros(len(chunk), dtype=np.float64)
            feats = []
            for i in range(len(chunk)):
                feat = enhanced_component_matrix(int(srcs[i]), float(times[i]), cmat[i], stats).astype(np.float32)
                if not use_candidate_prior:
                    feat[:, PRIOR_IDX] = 0.0
                feats.append(feat)
            feat_all = np.concatenate(feats, axis=0)
            feat_all = ((feat_all - mean) / std).astype(np.float32)
            raw_all = model(jt.array(feat_all)).numpy().reshape(len(chunk), len(cand_cols))
            for raw in raw_all:
                prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/enhanced_jittor_mlp")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_enhanced_jittor_rank")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "row_sigmoid", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--max-history-per-src", type=int, default=30)
    parser.add_argument("--max-pairs-per-src", type=int, default=30)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    jt.flags.use_cuda = 1 if args.use_cuda else 0
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
