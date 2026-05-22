from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simple_heuristic import COMPONENT_NAMES, DEFAULT_WEIGHTS, build_simple_stats, history_only, score_candidates
from track1_dynamic_rec.metrics import ranking_metrics
from track1_dynamic_rec.train_utils import save_json


def _candidate_cols(columns) -> list[str]:
    cols = [c for c in columns if str(c).lower().startswith("c") and str(c)[1:].isdigit()]
    return sorted(cols, key=lambda x: int(str(x)[1:]))


def _split_train_val(train: pd.DataFrame, dataset: str, val_ratio: float, max_val_events: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "dataset2" and "split" in train.columns:
        hist = history_only(train, include_valid_history=False)
        split = train["split"]
        val = train[split.astype(int) == 1].copy() if split.dtype.kind in {"i", "u", "f"} else train[split.astype(str).str.strip().isin(["1", "valid", "val", "validation"])].copy()
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
            bucket = by_src.setdefault(int(s), [])
            bucket.extend(int(x) for x in row)
    unique = np.unique(np.concatenate(values).astype(np.int64))
    by_src_arr = {k: np.unique(np.asarray(v, dtype=np.int64)) for k, v in by_src.items()}
    return unique, by_src_arr


def _make_candidates(
    hist: pd.DataFrame,
    val: pd.DataFrame,
    test_pool: np.ndarray,
    by_src: dict[int, np.ndarray],
    num_negatives: int,
    seed: int,
) -> pd.DataFrame:
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
        negs: list[int] = []
        blocked = set(known_by_src.get(src, set()))
        blocked.add(dst)

        src_pool = by_src.get(src)
        if src_pool is not None and len(src_pool):
            order = rng.permutation(len(src_pool))
            for j in order:
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
        for rank, ndst in enumerate(negs[:num_negatives], start=2):
            query_rows.append({"src": src, "dst": int(ndst), "time": t, "label": 0.0, "query_id": qid})
        order = rng.permutation(len(query_rows))
        for rank, pos in enumerate(order, start=1):
            item = query_rows[int(pos)]
            item["candidate_rank"] = rank
            rows.append(item)
    return pd.DataFrame(rows)


def _rank01(score: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(score))
    return order.astype(np.float32) / max(1, len(score) - 1)


def score_frame(candidates: pd.DataFrame, stats, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(len(candidates), dtype=np.float64)
    for _, idx in candidates.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        g = candidates.iloc[idx]
        raw = score_candidates(int(g["src"].iloc[0]), float(g["time"].iloc[0]), g["dst"].to_numpy(np.int64), stats, weights)
        out[idx] = _rank01(raw)
    return out


def metric(candidates: pd.DataFrame, score: np.ndarray) -> dict:
    pred = candidates.copy()
    pred["score"] = score
    return ranking_metrics(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--output-dir", type=str, default="outputs/simple_heuristic_hard_tuned")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-val-events", type=int, default=2000)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_root) / args.dataset
    train = pd.read_csv(data_dir / "train.csv")
    hist, val = _split_train_val(train, args.dataset, args.val_ratio, args.max_val_events)
    print(json.dumps({"dataset": args.dataset, "history_edges": len(hist), "val_edges": len(val)}, ensure_ascii=False))
    test_pool, by_src = _test_candidate_sources(data_dir / "test.csv")
    candidates = _make_candidates(hist, val, test_pool, by_src, args.negatives, args.seed)
    print(json.dumps({"candidates": len(candidates), "queries": candidates["query_id"].nunique(), "test_pool": len(test_pool)}, ensure_ascii=False))
    stats = build_simple_stats(hist)
    rng = np.random.RandomState(args.seed)
    best_w = DEFAULT_WEIGHTS.copy()
    best_score = score_frame(candidates, stats, best_w)
    best_metrics = metric(candidates, best_score)
    rows = [{"trial": -1, "kind": "default", **best_metrics}]
    for trial in range(args.trials):
        if trial < args.trials // 2:
            w = best_w * rng.lognormal(0.0, 0.25, size=len(best_w)) + rng.normal(0.0, 0.08, size=len(best_w))
        else:
            w = DEFAULT_WEIGHTS * rng.lognormal(0.0, 0.7, size=len(best_w)) + rng.normal(0.0, 0.25, size=len(best_w))
        score = score_frame(candidates, stats, w)
        m = metric(candidates, score)
        rows.append({"trial": trial, "kind": "random", **m})
        key = m.get("mrr", m.get("auc", -1e9))
        if key > best_metrics.get("mrr", best_metrics.get("auc", -1e9)):
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
    }
    save_json(output_dir / f"{args.dataset}_simple_heuristic.json", out)
    pd.DataFrame(rows).to_csv(output_dir / f"{args.dataset}_hard_tuning_log.csv", index=False)
    print(json.dumps({"best": best_metrics, "weights": best_w.tolist()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
