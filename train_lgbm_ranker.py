from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from make_assoc_result_zip import assoc_component_matrix, build_assoc_stats
from make_temporal_result_zip import _build_stats as build_temporal_stats
from make_temporal_result_zip import _parse_csv_floats, temporal_component_matrix
from sequential_heuristic import build_sequential_stats, sequential_component_matrix
from track1_dynamic_rec.train_utils import save_json, set_seed
from tune_sequential_hard import _make_candidates, _test_candidate_sources


def _split_internal_holdout(train: pd.DataFrame, val_ratio: float, max_val_events: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in train.columns:
        train = train[train["split"].astype(int) == 0].copy()
    train = train.sort_values("time").reset_index(drop=True)
    cut = max(1, min(len(train) - 1, int(len(train) * (1.0 - val_ratio))))
    hist = train.iloc[:cut].copy()
    val = train.iloc[cut:].copy()
    if max_val_events > 0 and len(val) > max_val_events:
        val = val.sort_values("time").tail(max_val_events).copy()
    return hist[["src", "dst", "time"]].copy(), val[["src", "dst", "time"]].copy()


def _feature_frame(
    candidates: pd.DataFrame,
    seq_stats,
    temporal_stats,
    assoc_stats,
    windows: list[float],
    taus: list[float],
    decay_cap: int,
    assoc_history_score: int,
) -> np.ndarray:
    width = 13 + 6 + len(windows) * 2 + len(taus) + 8
    out = np.zeros((len(candidates), width), dtype=np.float32)
    for _, idx in candidates.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        g = candidates.iloc[idx]
        src = int(g["src"].iloc[0])
        t = float(g["time"].iloc[0])
        dst = g["dst"].to_numpy(np.int64)
        sfeat = sequential_component_matrix(src, t, dst, seq_stats)
        tfeat = temporal_component_matrix(src, t, dst, temporal_stats, windows, taus, decay_cap)
        afeat = assoc_component_matrix(src, dst, assoc_stats, assoc_history_score)
        out[idx] = np.concatenate([sfeat, tfeat, afeat], axis=1).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--dataset", default="dataset2", choices=["dataset2"])
    parser.add_argument("--output-dir", default="outputs/lgbm_ranker_split0_internal")
    parser.add_argument("--val-ratio", type=float, default=0.14)
    parser.add_argument("--max-val-events", type=int, default=70000)
    parser.add_argument("--negatives", type=int, default=79)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--seq-history", type=int, default=22)
    parser.add_argument("--seq-pairs", type=int, default=24)
    parser.add_argument("--assoc-context", type=int, default=10)
    parser.add_argument("--assoc-history", type=int, default=32)
    parser.add_argument("--assoc-score-history", type=int, default=24)
    parser.add_argument("--assoc-min-count", type=float, default=1.2)
    parser.add_argument("--windows", default="604800,1209600,2592000,7776000,15552000,31536000")
    parser.add_argument("--taus", default="604800,1209600,2592000,7776000,15552000")
    parser.add_argument("--decay-cap", type=int, default=384)
    parser.add_argument("--num-boost-round", type=int, default=550)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--min-data-in-leaf", type=int, default=80)
    parser.add_argument("--feature-fraction", type=float, default=0.92)
    parser.add_argument("--bagging-fraction", type=float, default=0.88)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l2", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_root) / args.dataset
    train = pd.read_csv(data_dir / "train.csv")
    hist, val = _split_internal_holdout(train, args.val_ratio, args.max_val_events)
    test_pool, by_src = _test_candidate_sources(data_dir / "test.csv")
    candidates = _make_candidates(hist, val, test_pool, by_src, args.negatives, args.seed)
    candidate_pool = set(int(x) for x in test_pool)
    windows = _parse_csv_floats(args.windows)
    taus = _parse_csv_floats(args.taus)
    seq_stats = build_sequential_stats(hist, candidate_pool=candidate_pool, max_history_per_src=args.seq_history, max_pairs_per_src=args.seq_pairs)
    temporal_stats = build_temporal_stats(hist, candidate_pool=candidate_pool, keep_pair_times=True)
    assoc_stats = build_assoc_stats(
        hist,
        candidate_pool=candidate_pool,
        max_context=args.assoc_context,
        max_history_per_src=args.assoc_history,
        min_count=args.assoc_min_count,
    )
    print(json.dumps({"hist": len(hist), "holdout": len(val), "candidates": len(candidates)}, ensure_ascii=False), flush=True)

    feat = _feature_frame(candidates, seq_stats, temporal_stats, assoc_stats, windows, taus, args.decay_cap, args.assoc_score_history)
    mean = feat.mean(axis=0).astype(np.float32)
    std = feat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    feat = ((feat - mean) / std).astype(np.float32)
    labels = candidates["label"].to_numpy(np.float32)
    group = candidates.groupby("query_id", sort=False).size().to_numpy(np.int32)
    print(json.dumps({"feature_shape": list(feat.shape), "groups": int(len(group)), "group_min": int(group.min()), "group_max": int(group.max())}, ensure_ascii=False), flush=True)

    train_set = lgb.Dataset(feat, label=labels, group=group, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 10],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "label_gain": [0, 1],
        "num_threads": 0,
        "seed": args.seed,
        "verbosity": 1,
    }
    booster = lgb.train(params, train_set, num_boost_round=args.num_boost_round)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_dir / "dataset2_lgbm_ranker.txt"))
    save_json(
        output_dir / "dataset2_lgbm_ranker.json",
        {
            "args": vars(args),
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "windows": windows,
            "taus": taus,
            "feature_dim": int(feat.shape[1]),
            "num_rows": int(len(feat)),
            "num_groups": int(len(group)),
            "note": "LightGBM auxiliary ranker trained only on dataset2 split=0 internal holdout; Jittor baseline remains the main competition framework.",
        },
    )
    print(json.dumps({"wrote": str(output_dir), "best_iteration": int(booster.best_iteration or args.num_boost_round)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
