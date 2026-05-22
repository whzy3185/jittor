from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simple_heuristic import COMPONENT_NAMES, DEFAULT_WEIGHTS, build_simple_stats, component_matrix
from track1_dynamic_rec.data import describe_bundle, load_competition_data
from track1_dynamic_rec.metrics import ranking_metrics
from track1_dynamic_rec.train_utils import save_json


def _rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(score))
    return order.astype(np.float32) / max(1, len(score) - 1)


def score_frame(candidates: pd.DataFrame, stats, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(len(candidates), dtype=np.float64)
    for _, idx in candidates.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        g = candidates.iloc[idx]
        src = int(g["src"].iloc[0])
        t = float(g["time"].iloc[0])
        cands = g["dst"].to_numpy(np.int64)
        raw = component_matrix(src, t, cands, stats) @ weights
        out[idx] = _rank01(raw)
    return out


def shuffle_candidates_within_query(candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    parts = []
    for _, group in candidates.groupby("query_id", sort=False):
        order = rng.permutation(len(group))
        g = group.iloc[order].copy().reset_index(drop=True)
        g["candidate_rank"] = np.arange(1, len(g) + 1, dtype=np.int64)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def metric(candidates: pd.DataFrame, score: np.ndarray) -> dict:
    pred = candidates.copy()
    pred["score"] = score
    return ranking_metrics(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/simple_heuristic_tuned")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-negatives", type=int, default=50)
    parser.add_argument("--max-val-events", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    bundle = load_competition_data(
        data_dir,
        val_ratio=args.val_ratio,
        num_val_negatives=args.val_negatives,
        seed=args.seed,
        max_val_events=args.max_val_events,
        load_test_candidates=False,
        map_test_candidates=False,
    )
    print(describe_bundle(bundle))
    if bundle.val_candidates is None or len(bundle.val_candidates) == 0:
        raise RuntimeError("No validation candidates available")
    candidates = shuffle_candidates_within_query(bundle.val_candidates, args.seed + 17)
    stats = build_simple_stats(bundle.train_edges)
    rng = np.random.RandomState(args.seed)
    best_w = DEFAULT_WEIGHTS.copy()
    best_score = score_frame(candidates, stats, best_w)
    best_metrics = metric(candidates, best_score)
    rows = [{"trial": -1, "kind": "default", **best_metrics}]
    for trial in range(args.trials):
        if trial < args.trials // 2:
            w = best_w * rng.lognormal(mean=0.0, sigma=0.35, size=len(best_w))
            w += rng.normal(0.0, 0.15, size=len(best_w))
        else:
            w = DEFAULT_WEIGHTS * rng.lognormal(mean=0.0, sigma=0.75, size=len(best_w))
            w += rng.normal(0.0, 0.35, size=len(best_w))
        score = score_frame(candidates, stats, w)
        m = metric(candidates, score)
        rows.append({"trial": trial, "kind": "random", **m})
        key = m.get("mrr", m.get("auc", -1e9))
        best_key = best_metrics.get("mrr", best_metrics.get("auc", -1e9))
        if key > best_key:
            best_w = w
            best_metrics = m
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": args.dataset,
        "component_names": COMPONENT_NAMES,
        "weights": best_w.astype(float).tolist(),
        "default_weights": DEFAULT_WEIGHTS.astype(float).tolist(),
        "best_metrics": best_metrics,
        "args": vars(args),
        "notes": bundle.notes,
    }
    save_json(output_dir / f"{args.dataset}_simple_heuristic.json", out)
    pd.DataFrame(rows).to_csv(output_dir / f"{args.dataset}_tuning_log.csv", index=False)
    print(json.dumps({"best": best_metrics, "weights": best_w.tolist()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
