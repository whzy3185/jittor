from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from make_assoc_result_zip import assoc_component_matrix, build_assoc_stats
from make_combo_jittor_result_zip import _candidate_pool, _prob
from make_temporal_result_zip import _build_stats as build_temporal_stats
from make_temporal_result_zip import temporal_component_matrix
from sequential_heuristic import build_sequential_stats, sequential_component_matrix
from simple_heuristic import history_only
from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.train_utils import load_json


def _feature_matrix_for_row(
    src: int,
    time_value: float,
    candidates: np.ndarray,
    seq_stats,
    temporal_stats,
    assoc_stats,
    windows: list[float],
    taus: list[float],
    decay_cap: int,
    assoc_history_score: int,
) -> np.ndarray:
    sfeat = sequential_component_matrix(src, time_value, candidates, seq_stats)
    tfeat = temporal_component_matrix(src, time_value, candidates, temporal_stats, windows, taus, decay_cap)
    afeat = assoc_component_matrix(src, candidates, assoc_stats, assoc_history_score)
    return np.concatenate([sfeat, tfeat, afeat], axis=1).astype(np.float32)


def _write_dataset2(
    data_root: Path,
    model_dir: Path,
    output_csv: Path,
    include_valid_history: bool,
    mode: str,
    chunksize: int,
) -> int:
    meta = load_json(model_dir / "dataset2_lgbm_ranker.json")
    args = meta["args"]
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    windows = [float(x) for x in meta["windows"]]
    taus = [float(x) for x in meta["taus"]]
    booster = lgb.Booster(model_file=str(model_dir / "dataset2_lgbm_ranker.txt"))

    data_dir = data_root / "dataset2"
    train = history_only(pd.read_csv(data_dir / "train.csv"), include_valid_history=include_valid_history)
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    candidate_pool = _candidate_pool(test_path, cand_cols)
    seq_stats = build_sequential_stats(
        train,
        candidate_pool=candidate_pool,
        max_history_per_src=int(args["seq_history"]),
        max_pairs_per_src=int(args["seq_pairs"]),
    )
    temporal_stats = build_temporal_stats(train, candidate_pool=candidate_pool, keep_pair_times=True)
    assoc_stats = build_assoc_stats(
        train,
        candidate_pool=candidate_pool,
        max_context=int(args["assoc_context"]),
        max_history_per_src=int(args["assoc_history"]),
        min_count=float(args["assoc_min_count"]),
    )
    print({"dataset": "dataset2", "history_edges": len(train), "candidate_pool": len(candidate_pool), "model_dir": str(model_dir)}, flush=True)

    rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunksize):
            srcs = chunk["src"].to_numpy(np.int64)
            times = chunk["time"].to_numpy(np.float64)
            mat = chunk[cand_cols].to_numpy(np.int64)
            feats = []
            for src, t, row in zip(srcs, times, mat):
                feat = _feature_matrix_for_row(
                    int(src),
                    float(t),
                    row,
                    seq_stats,
                    temporal_stats,
                    assoc_stats,
                    windows,
                    taus,
                    int(args["decay_cap"]),
                    int(args["assoc_score_history"]),
                )
                feats.append(((feat - mean) / std).astype(np.float32))
            batch_feat = np.concatenate(feats, axis=0).astype(np.float32)
            raw = booster.predict(batch_feat).reshape((len(feats), 100))
            for row in raw:
                prob = np.clip(_prob(row, mode, 1.0), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % 20000 < len(chunk):
                print({"dataset": "dataset2", "rows": rows}, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--baseline-dir", default="outputs/website_submission_121_major_teacher110_multi_replay_ensemble")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["rank", "row_sigmoid", "minmax"], default="rank")
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--chunksize", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.baseline_dir) / "dataset1.csv", output_dir / "dataset1.csv")
    rows = {
        "dataset1": sum(1 for _ in (output_dir / "dataset1.csv").open("r", encoding="utf-8")),
        "dataset2": _write_dataset2(Path(args.data_root), Path(args.model_dir), output_dir / "dataset2.csv", args.include_valid_history, args.mode, args.chunksize),
    }
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows})


if __name__ == "__main__":
    main()
