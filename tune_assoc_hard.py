from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_assoc_result_zip import assoc_component_matrix, build_assoc_stats
from simple_heuristic import history_only
from track1_dynamic_rec.metrics import ranking_metrics
from track1_dynamic_rec.train_utils import save_json
from tune_sequential_hard import _make_candidates, _test_candidate_sources


DEFAULT_ASSOC_WEIGHTS = np.asarray([2.0, 1.5, 1.2, 0.4, 1.1, 2.5, 1.0, 0.0], dtype=np.float64)
ASSOC_FEATURE_NAMES = [
    "assoc_total",
    "assoc_best",
    "assoc_last",
    "assoc_hit_count",
    "assoc_pmi_total",
    "pair_log",
    "pair_share",
    "candidate_prior",
]


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


def feature_frame(candidates: pd.DataFrame, stats, max_history_score: int) -> np.ndarray:
    out = np.zeros((len(candidates), len(ASSOC_FEATURE_NAMES)), dtype=np.float64)
    for _, idx in candidates.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        g = candidates.iloc[idx]
        out[idx] = assoc_component_matrix(int(g["src"].iloc[0]), g["dst"].to_numpy(np.int64), stats, max_history_score)
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
    parser.add_argument("--output-dir", type=str, default="outputs/assoc_heuristic_hard")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-val-events", type=int, default=3000)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--trials", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context", type=int, default=8)
    parser.add_argument("--max-history-per-src", type=int, default=24)
    parser.add_argument("--max-history-score", type=int, default=16)
    parser.add_argument("--min-count", type=float, default=1.5)
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset
    train = pd.read_csv(data_dir / "train.csv")
    hist, val = _split_train_val(train, args.dataset, args.val_ratio, args.max_val_events)
    test_pool, by_src = _test_candidate_sources(data_dir / "test.csv")
    candidates = _make_candidates(hist, val, test_pool, by_src, args.negatives, args.seed)
    stats = build_assoc_stats(
        hist,
        candidate_pool=set(int(x) for x in test_pool),
        max_context=args.max_context,
        max_history_per_src=args.max_history_per_src,
        min_count=args.min_count,
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "history_edges": len(hist),
                "val_edges": len(val),
                "candidates": len(candidates),
                "assoc_pairs": len(stats.assoc),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    features = feature_frame(candidates, stats, args.max_history_score)
    print(json.dumps({"feature_matrix": list(features.shape)}, ensure_ascii=False), flush=True)

    rng = np.random.RandomState(args.seed)
    best_w = DEFAULT_ASSOC_WEIGHTS.copy()
    best_metrics = metric(candidates, score_from_features(candidates, features, best_w))
    rows = [{"trial": -1, "kind": "default", **best_metrics}]
    for trial in range(args.trials):
        if trial < args.trials // 2:
            w = best_w * rng.lognormal(0.0, 0.30, size=len(best_w)) + rng.normal(0.0, 0.08, size=len(best_w))
        else:
            w = DEFAULT_ASSOC_WEIGHTS * rng.lognormal(0.0, 0.80, size=len(best_w)) + rng.normal(0.0, 0.25, size=len(best_w))
        w = np.clip(w, -5.0, 30.0)
        m = metric(candidates, score_from_features(candidates, features, w))
        rows.append({"trial": trial, "kind": "random", **m})
        key = m.get("mrr", m.get("auc", -1e9))
        if key > best_metrics.get("mrr", best_metrics.get("auc", -1e9)):
            best_w = w
            best_metrics = m
            print(json.dumps({"trial": trial, "best": best_metrics}, ensure_ascii=False), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / f"{args.dataset}_assoc_heuristic.json",
        {
            "dataset": args.dataset,
            "feature_names": ASSOC_FEATURE_NAMES,
            "weights": best_w.astype(float).tolist(),
            "default_weights": DEFAULT_ASSOC_WEIGHTS.astype(float).tolist(),
            "best_metrics": best_metrics,
            "args": vars(args),
        },
    )
    pd.DataFrame(rows).to_csv(output_dir / f"{args.dataset}_assoc_tuning_log.csv", index=False)
    print(json.dumps({"best": best_metrics, "weights": dict(zip(ASSOC_FEATURE_NAMES, best_w.astype(float).tolist()))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
