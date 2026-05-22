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
from make_combo_jittor_result_zip import _candidate_pool, _prob
from make_temporal_result_zip import _build_stats as build_temporal_stats
from make_temporal_result_zip import temporal_component_matrix
from sequential_heuristic import build_sequential_stats, sequential_component_matrix
from simple_heuristic import history_only
from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.train_utils import load_json
from train_combo_mlp_jittor import ComboMLPRanker


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
    temperature: float,
    chunksize: int,
) -> int:
    meta = load_json(model_dir / "dataset2_combo_mlp.json")
    args = meta["args"]
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    windows = [float(x) for x in meta["windows"]]
    taus = [float(x) for x in meta["taus"]]
    model = ComboMLPRanker(int(meta["feature_dim"]), int(args["hidden_dim"]), float(args["dropout"]))
    model.load_state_dict(jt.load(str(model_dir / meta["checkpoint"])))
    model.eval()

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
    print(
        {
            "dataset": "dataset2",
            "history_edges": len(train),
            "candidate_pool": len(candidate_pool),
            "model_dir": str(model_dir),
            "mode": mode,
        },
        flush=True,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
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
                feat = ((feat - mean) / std).astype(np.float32)
                feats.append(feat)
            batch_feat = np.concatenate(feats, axis=0).astype(np.float32)
            batch_raw = model(jt.array(batch_feat)).numpy().reshape((len(feats), 100))
            for raw in batch_raw:
                prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % 20000 < len(chunk):
                print({"dataset": "dataset2", "rows": rows}, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/combo_mlp_jittor_split0_internal")
    parser.add_argument("--baseline-dir", type=str, default="outputs/website_submission_89_major_teacher88_online_top4q80_40")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "row_sigmoid", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--chunksize", type=int, default=1024)
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
            mode=args.mode,
            temperature=args.temperature,
            chunksize=args.chunksize,
        ),
    }
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows})


if __name__ == "__main__":
    main()
