from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from track1_dynamic_rec.data import describe_bundle, load_competition_data
from track1_dynamic_rec.features import FEATURE_NAMES, build_candidate_features, resolve_time_window
from track1_dynamic_rec.metrics import ranking_metrics
from track1_dynamic_rec.train_utils import save_json


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float32) / max(1, len(values) - 1)


def _group_rank01(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.float32)
    tmp = df[["query_id"]].copy()
    tmp["score"] = score
    for _, idx in tmp.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        out[idx] = _rank01(score[idx])
    return out


def _metric_for_score(candidates: pd.DataFrame, score: np.ndarray) -> dict:
    pred = candidates.copy()
    pred["score"] = _group_rank01(pred, score)
    return ranking_metrics(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/heuristic_linear_tuned")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-negatives", type=int, default=50)
    parser.add_argument("--max-val-events", type=int, default=5000)
    parser.add_argument("--max-test-queries", type=int, default=0)
    parser.add_argument("--window", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    bundle = load_competition_data(
        data_dir,
        val_ratio=args.val_ratio,
        num_val_negatives=args.val_negatives,
        seed=args.seed,
        max_val_events=args.max_val_events,
        max_test_queries=args.max_test_queries,
        load_test_candidates=False,
        map_test_candidates=False,
    )
    print(describe_bundle(bundle))
    if bundle.val_candidates is None or len(bundle.val_candidates) == 0:
        raise RuntimeError("No validation candidates available for tuning.")
    window = resolve_time_window(bundle.train_edges, args.window)
    feat, heuristic = build_candidate_features(bundle.val_candidates, bundle.train_edges, window=window)
    X = np.column_stack([feat, heuristic.astype(np.float32)])
    names = FEATURE_NAMES + ["base_heuristic"]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    base = np.zeros(Xn.shape[1], dtype=np.float64)
    base[-1] = 1.0
    candidates = bundle.val_candidates
    best_w = base.copy()
    best_score = Xn @ best_w
    best_metrics = _metric_for_score(candidates, best_score)
    rows = [{"trial": -1, "kind": "base", **best_metrics}]

    rng = np.random.RandomState(args.seed)
    scales = [0.05, 0.1, 0.2, 0.4, 0.8]
    for trial in range(args.trials):
        scale = scales[min(len(scales) - 1, trial * len(scales) // max(1, args.trials))]
        w = best_w + rng.normal(0.0, scale, size=best_w.shape)
        if rng.rand() < 0.25:
            w = rng.normal(0.0, scale * 2.0, size=best_w.shape)
            w[-1] += 1.0
        score = Xn @ w
        metrics = _metric_for_score(candidates, score)
        rows.append({"trial": trial, "kind": "random", **metrics})
        key = metrics.get("mrr", metrics.get("auc", -1e9))
        best_key = best_metrics.get("mrr", best_metrics.get("auc", -1e9))
        if key > best_key:
            best_w = w
            best_metrics = metrics

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "window": float(window),
        "feature_names": names,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "weights": best_w.astype(float).tolist(),
        "best_metrics": best_metrics,
        "args": vars(args),
        "notes": bundle.notes,
    }
    save_json(output_dir / f"{args.dataset}_linear_heuristic.json", out)
    pd.DataFrame(rows).to_csv(output_dir / f"{args.dataset}_tuning_log.csv", index=False)
    print(json.dumps({"best": best_metrics, "path": str(output_dir / f"{args.dataset}_linear_heuristic.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
