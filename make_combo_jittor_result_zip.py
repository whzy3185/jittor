from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt

from make_assoc_result_zip import assoc_component_matrix, build_assoc_stats
from make_temporal_result_zip import _build_stats as build_temporal_stats
from make_temporal_result_zip import temporal_component_matrix
from sequential_heuristic import build_sequential_stats, sequential_component_matrix
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


def _write_dataset2(
    data_root: Path,
    model_dir: Path,
    output_csv: Path,
    include_valid_history: bool,
    chunk_size: int,
    mode: str,
    temperature: float,
) -> int:
    dataset = "dataset2"
    meta = load_json(model_dir / "dataset2_combo_jittor.json")
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    weights = np.asarray(meta["weights"], dtype=np.float32)
    bias = float(meta.get("bias", 0.0))
    args = meta["args"]
    windows = [float(x) for x in meta["windows"]]
    taus = [float(x) for x in meta["taus"]]
    data_dir = data_root / dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=include_valid_history)
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    pool = _candidate_pool(test_path, cand_cols)
    seq_stats = build_sequential_stats(
        train,
        candidate_pool=pool,
        max_history_per_src=int(args["seq_history"]),
        max_pairs_per_src=int(args["seq_pairs"]),
    )
    temporal_stats = build_temporal_stats(train, candidate_pool=pool, keep_pair_times=True)
    assoc_stats = build_assoc_stats(
        train,
        candidate_pool=pool,
        max_context=int(args["assoc_context"]),
        max_history_per_src=int(args["assoc_history"]),
        min_count=float(args["assoc_min_count"]),
    )
    print(
        {
            "dataset": dataset,
            "history_edges": len(train),
            "candidate_pool": len(pool),
            "transitions": len(seq_stats.trans_count),
            "assoc_pairs": len(assoc_stats.assoc),
            "feature_dim": len(weights),
        },
        flush=True,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    w_jt = jt.array(weights.reshape((-1, 1)))
    b_jt = jt.array(np.asarray([bias], dtype=np.float32))
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            srcs = chunk["src"].to_numpy(np.int64)
            times = chunk["time"].to_numpy(np.float64)
            mat = chunk[cand_cols].to_numpy(np.int64)
            feats = []
            for src, t, dst in zip(srcs, times, mat):
                sfeat = sequential_component_matrix(int(src), float(t), dst, seq_stats)
                tfeat = temporal_component_matrix(
                    int(src),
                    float(t),
                    dst,
                    temporal_stats,
                    windows,
                    taus,
                    int(args["decay_cap"]),
                )
                afeat = assoc_component_matrix(int(src), dst, assoc_stats, int(args["assoc_score_history"]))
                feats.append(np.concatenate([sfeat, tfeat, afeat], axis=1).astype(np.float32))
            feat_all = np.concatenate(feats, axis=0)
            feat_all = ((feat_all - mean) / std).astype(np.float32)
            raw = (jt.array(feat_all).matmul(w_jt) + b_jt).numpy().reshape((len(chunk), len(cand_cols)))
            for row in raw:
                prob = np.clip(_prob(row, mode, temperature), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/combo_jittor")
    parser.add_argument("--baseline-dir", type=str, default="outputs/website_submission_blend_dsaware_seq")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_combo_jittor")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "row_sigmoid", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.baseline_dir) / "dataset1.csv", output_dir / "dataset1.csv")
    rows = {
        "dataset1": sum(1 for _ in (output_dir / "dataset1.csv").open("r", encoding="utf-8")),
        "dataset2": _write_dataset2(
            Path(args.data_root),
            Path(args.model_dir),
            output_dir / "dataset2.csv",
            include_valid_history=args.include_valid_history,
            chunk_size=args.chunk_size,
            mode=args.mode,
            temperature=args.temperature,
        ),
    }
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode})


if __name__ == "__main__":
    main()
