from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sequential_heuristic import (
    SEQUENTIAL_COMPONENT_NAMES,
    SEQUENTIAL_DEFAULT_WEIGHTS,
    build_sequential_stats,
    sequential_component_matrix,
)
from simple_heuristic import history_only
from track1_dynamic_rec.metrics import ranking_metrics
from track1_dynamic_rec.train_utils import save_json


def _candidate_cols(columns) -> list[str]:
    cols = [c for c in columns if str(c).lower().startswith("c") and str(c)[1:].isdigit()]
    return sorted(cols, key=lambda x: int(str(x)[1:]))


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


def _test_candidate_sources(test_path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    header = pd.read_csv(test_path, nrows=0)
    cand_cols = _candidate_cols(header.columns)
    values = []
    by_src: dict[int, list[int]] = {}
    for chunk in pd.read_csv(test_path, usecols=["src"] + cand_cols, chunksize=8192):
        src = chunk["src"].to_numpy(np.int64)
        mat = chunk[cand_cols].to_numpy(np.int64)
        values.append(mat.reshape(-1))
        for s, row in zip(src, mat):
            by_src.setdefault(int(s), []).extend(int(x) for x in row)
    return np.unique(np.concatenate(values).astype(np.int64)), {k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in by_src.items()}


def _make_candidates(hist: pd.DataFrame, val: pd.DataFrame, test_pool: np.ndarray, by_src: dict[int, np.ndarray], num_negatives: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dst_counts = hist["dst"].value_counts()
    popular = dst_counts.index.to_numpy(np.int64)
    pop_prob = np.power(dst_counts.to_numpy(np.float64), 0.75)
    pop_prob = pop_prob / pop_prob.sum()
    known_by_src = hist.groupby("src")["dst"].agg(lambda x: set(int(v) for v in x)).to_dict()
    rows = []
    val = val.sort_values("time").reset_index(drop=True)
    for i, row in enumerate(val.itertuples(index=False)):
        src, dst, t = int(row.src), int(row.dst), float(row.time)
        qid = f"valid_{i}"
        query_rows = [{"src": src, "dst": dst, "time": t, "label": 1.0, "query_id": qid}]
        blocked = set(known_by_src.get(src, set()))
        blocked.add(dst)
        negs: list[int] = []
        src_pool = by_src.get(src)
        if src_pool is not None and len(src_pool):
            for j in rng.permutation(len(src_pool)):
                x = int(src_pool[j])
                if x not in blocked and x not in negs:
                    negs.append(x)
                if len(negs) >= num_negatives // 2:
                    break
        tries = 0
        while len(negs) < num_negatives and tries < num_negatives * 200 + 500:
            tries += 1
            if rng.rand() < 0.65 and len(popular):
                x = int(rng.choice(popular, p=pop_prob))
            else:
                x = int(test_pool[rng.randint(0, len(test_pool))])
            if x in blocked or x in negs:
                continue
            negs.append(x)
        for ndst in negs[:num_negatives]:
            query_rows.append({"src": src, "dst": int(ndst), "time": t, "label": 0.0, "query_id": qid})
        for rank, pos in enumerate(rng.permutation(len(query_rows)), start=1):
            item = query_rows[int(pos)]
            item["candidate_rank"] = rank
            rows.append(item)
    return pd.DataFrame(rows)


def _rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(score))
    return order.astype(np.float64) / max(1, len(score) - 1)


def feature_frame(candidates: pd.DataFrame, stats) -> np.ndarray:
    out = np.zeros((len(candidates), len(SEQUENTIAL_COMPONENT_NAMES)), dtype=np.float64)
    for _, idx in candidates.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        g = candidates.iloc[idx]
        out[idx] = sequential_component_matrix(int(g["src"].iloc[0]), float(g["time"].iloc[0]), g["dst"].to_numpy(np.int64), stats)
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
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--output-dir", type=str, default="outputs/sequential_heuristic_hard")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-val-events", type=int, default=3000)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history-per-src", type=int, default=30)
    parser.add_argument("--max-pairs-per-src", type=int, default=30)
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset
    train = pd.read_csv(data_dir / "train.csv")
    hist, val = _split_train_val(train, args.dataset, args.val_ratio, args.max_val_events)
    test_pool, by_src = _test_candidate_sources(data_dir / "test.csv")
    candidates = _make_candidates(hist, val, test_pool, by_src, args.negatives, args.seed)
    stats = build_sequential_stats(
        hist,
        candidate_pool=set(int(x) for x in test_pool),
        max_history_per_src=args.max_history_per_src,
        max_pairs_per_src=args.max_pairs_per_src,
    )
    print(json.dumps({"dataset": args.dataset, "history_edges": len(hist), "val_edges": len(val), "candidates": len(candidates), "transitions": len(stats.trans_count)}, ensure_ascii=False))
    features = feature_frame(candidates, stats)
    print(json.dumps({"feature_matrix": list(features.shape)}, ensure_ascii=False), flush=True)

    rng = np.random.RandomState(args.seed)
    best_w = SEQUENTIAL_DEFAULT_WEIGHTS.copy()
    best_metrics = metric(candidates, score_from_features(candidates, features, best_w))
    rows = [{"trial": -1, "kind": "default", **best_metrics}]
    for trial in range(args.trials):
        if trial < args.trials // 2:
            w = best_w * rng.lognormal(0.0, 0.20, size=len(best_w)) + rng.normal(0.0, 0.05, size=len(best_w))
        else:
            w = SEQUENTIAL_DEFAULT_WEIGHTS * rng.lognormal(0.0, 0.65, size=len(best_w)) + rng.normal(0.0, 0.20, size=len(best_w))
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
    save_json(
        output_dir / f"{args.dataset}_sequential_heuristic.json",
        {
            "dataset": args.dataset,
            "component_names": SEQUENTIAL_COMPONENT_NAMES,
            "weights": best_w.astype(float).tolist(),
            "default_weights": SEQUENTIAL_DEFAULT_WEIGHTS.astype(float).tolist(),
            "best_metrics": best_metrics,
            "args": vars(args),
        },
    )
    pd.DataFrame(rows).to_csv(output_dir / f"{args.dataset}_sequential_tuning_log.csv", index=False)
    print(json.dumps({"best": best_metrics, "weights": dict(zip(SEQUENTIAL_COMPONENT_NAMES, best_w.astype(float).tolist()))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
