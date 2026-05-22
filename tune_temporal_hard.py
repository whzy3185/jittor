from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_temporal_result_zip import _build_stats, _parse_csv_floats, temporal_component_matrix
from simple_heuristic import history_only
from track1_dynamic_rec.metrics import ranking_metrics
from track1_dynamic_rec.train_utils import save_json
from tune_sequential_hard import _make_candidates, _test_candidate_sources


DEFAULT_TEMPORAL_WEIGHTS = np.asarray(
    [4.0, 0.05, 1.2, 5.0, 0.5, 0.0, 0.8, 1.0, 1.2, 1.0, 1.2, 1.4, 1.4, 1.0, 1.2, 1.0, 0.8],
    dtype=np.float64,
)


def _split_train_val(train: pd.DataFrame, dataset: str, val_ratio: float, max_val_events: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "dataset2" and "split" in train.columns:
        hist = history_only(train, include_valid_history=False)
        val = train[train["split"].astype(int) == 1].copy()
    else:
        train = train.sort_values("time").reset_index(drop=True)
        cut = max(1, int(len(train) * (1.0 - val_ratio)))
        hist = train.iloc[:cut].copy()
        val = train.iloc[cut:].copy()
    if max_val_events > 0 and len(val) > max_val_events:
        val = val.sort_values("time").tail(max_val_events).copy()
    return hist[["src", "dst", "time"]].copy(), val[["src", "dst", "time"]].copy()


def _rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(score))
    return order.astype(np.float64) / max(1, len(score) - 1)


def feature_frame(candidates: pd.DataFrame, stats, windows: list[float], taus: list[float], decay_cap: int) -> np.ndarray:
    width = 6 + len(windows) * 2 + len(taus)
    out = np.zeros((len(candidates), width), dtype=np.float64)
    for _, idx in candidates.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        g = candidates.iloc[idx]
        out[idx] = temporal_component_matrix(
            int(g["src"].iloc[0]),
            float(g["time"].iloc[0]),
            g["dst"].to_numpy(np.int64),
            stats,
            windows,
            taus,
            decay_cap,
        )
    return out


def score_from_features(candidates: pd.DataFrame, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    raw_all = features @ weights
    out = np.zeros(len(candidates), dtype=np.float64)
    for _, idx in candidates.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        out[idx] = _rank01(raw_all[idx])
    return out


def metric(candidates: pd.DataFrame, score: np.ndarray) -> dict:
    pred = candidates.copy()
    pred["score"] = score
    return ranking_metrics(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset2", choices=["dataset1", "dataset2"])
    parser.add_argument("--output-dir", type=str, default="outputs/temporal_heuristic_hard")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-val-events", type=int, default=3000)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=str, default="604800,2592000,7776000,31536000")
    parser.add_argument("--taus", type=str, default="604800,2592000,7776000")
    parser.add_argument("--decay-cap", type=int, default=256)
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset
    train = pd.read_csv(data_dir / "train.csv")
    hist, val = _split_train_val(train, args.dataset, args.val_ratio, args.max_val_events)
    test_pool, by_src = _test_candidate_sources(data_dir / "test.csv")
    candidates = _make_candidates(hist, val, test_pool, by_src, args.negatives, args.seed)
    windows = _parse_csv_floats(args.windows)
    taus = _parse_csv_floats(args.taus)
    stats = _build_stats(hist, candidate_pool=set(int(x) for x in test_pool), keep_pair_times=True)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "history_edges": len(hist),
                "val_edges": len(val),
                "candidates": len(candidates),
                "dst_series": len(stats.dst_times),
                "pair_series": len(stats.pair_times),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    features = feature_frame(candidates, stats, windows, taus, args.decay_cap)
    print(json.dumps({"feature_matrix": list(features.shape)}, ensure_ascii=False), flush=True)

    rng = np.random.RandomState(args.seed)
    best_w = DEFAULT_TEMPORAL_WEIGHTS.copy()
    best_metrics = metric(candidates, score_from_features(candidates, features, best_w))
    rows = [{"trial": -1, "kind": "default", **best_metrics}]
    for trial in range(args.trials):
        if trial < args.trials // 2:
            w = best_w * rng.lognormal(0.0, 0.28, size=len(best_w)) + rng.normal(0.0, 0.08, size=len(best_w))
        else:
            w = DEFAULT_TEMPORAL_WEIGHTS * rng.lognormal(0.0, 0.75, size=len(best_w)) + rng.normal(0.0, 0.20, size=len(best_w))
        w = np.clip(w, -5.0, 30.0)
        score = score_from_features(candidates, features, w)
        m = metric(candidates, score)
        rows.append({"trial": trial, "kind": "random", **m})
        key = m.get("mrr", m.get("auc", -1e9))
        if key > best_metrics.get("mrr", best_metrics.get("auc", -1e9)):
            best_w = w
            best_metrics = m
            print(json.dumps({"trial": trial, "best": best_metrics}, ensure_ascii=False), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = (
        ["pair_log", "dst_log", "pair_recency", "dst_recency", "src_recency", "candidate_prior"]
        + [f"dst_win_{int(w)}" for w in windows]
        + [f"pair_win_{int(w)}" for w in windows]
        + [f"dst_decay_{int(t)}" for t in taus]
    )
    save_json(
        output_dir / f"{args.dataset}_temporal_heuristic.json",
        {
            "dataset": args.dataset,
            "feature_names": feature_names,
            "weights": best_w.astype(float).tolist(),
            "default_weights": DEFAULT_TEMPORAL_WEIGHTS.astype(float).tolist(),
            "best_metrics": best_metrics,
            "windows": windows,
            "taus": taus,
            "args": vars(args),
        },
    )
    pd.DataFrame(rows).to_csv(output_dir / f"{args.dataset}_temporal_tuning_log.csv", index=False)
    print(json.dumps({"best": best_metrics, "weights": dict(zip(feature_names, best_w.astype(float).tolist()))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
