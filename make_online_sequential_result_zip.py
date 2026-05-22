from __future__ import annotations

import argparse
import csv
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from sequential_heuristic import SEQUENTIAL_DEFAULT_WEIGHTS, build_sequential_stats, score_sequential_candidates
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


def _push_recent(recent: np.ndarray | None, dst: int, limit: int) -> np.ndarray:
    dst = int(dst)
    out = [dst]
    if recent is not None:
        for x in recent:
            y = int(x)
            if y == dst:
                continue
            out.append(y)
            if len(out) >= limit:
                break
    return np.asarray(out[:limit], dtype=np.int64)


def _update_stats(stats, src: int, dst: int, time_value: float, recent_limit: int) -> None:
    src = int(src)
    dst = int(dst)
    time_value = float(time_value)
    simple = stats.enhanced.simple
    old_recent = stats.enhanced.recent_by_src.get(src)
    if old_recent is not None and len(old_recent) > 0:
        prev = int(old_recent[0])
        if prev != dst:
            stats.trans_count[(prev, dst)] = int(stats.trans_count.get((prev, dst), 0)) + 1
            stats.trans_out[prev] = int(stats.trans_out.get(prev, 0)) + 1
    simple.dst_pop[dst] = int(simple.dst_pop.get(dst, 0)) + 1
    simple.src_degree[src] = int(simple.src_degree.get(src, 0)) + 1
    key = (src, dst)
    simple.pair_count[key] = int(simple.pair_count.get(key, 0)) + 1
    simple.last_pair[key] = time_value
    simple.last_dst[dst] = time_value
    simple.last_src[src] = time_value
    stats.enhanced.recent_by_src[src] = _push_recent(old_recent, dst, recent_limit)


def _write_dataset(
    dataset: str,
    data_root: Path,
    model_dir: Path,
    output_csv: Path,
    include_valid_history: bool,
    mode: str,
    temperature: float,
    max_history_per_src: int,
    max_pairs_per_src: int,
    update_topk: int,
    margin_quantile: float,
) -> int:
    data_dir = data_root / dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=include_valid_history)
    test_path = data_dir / "test.csv"
    header = pd.read_csv(test_path, nrows=0).columns
    cand_cols = wide_candidate_columns(header)
    pool = _candidate_pool(test_path, cand_cols)
    stats = build_sequential_stats(
        train,
        candidate_pool=pool,
        max_history_per_src=max_history_per_src,
        max_pairs_per_src=max_pairs_per_src,
    )
    model_path = model_dir / f"{dataset}_sequential_heuristic.json"
    weights = np.asarray(load_json(model_path)["weights"], dtype=np.float64) if model_path.exists() else SEQUENTIAL_DEFAULT_WEIGHTS
    test = pd.read_csv(test_path)
    original_index = test.index.to_numpy(np.int64)
    test = test.assign(_row_id=original_index).sort_values(["time", "_row_id"]).reset_index(drop=True)
    out_rows: list[list[str] | None] = [None] * len(test)
    update_buffer: list[tuple[int, int, float]] = []
    margin_values: list[float] = []
    rows = 0
    last_time = None
    pseudo_edges = 0
    for _, row_obj in test.iterrows():
        time_value = float(row_obj["time"])
        if last_time is None:
            last_time = time_value
        if time_value != last_time:
            if margin_values and 0.0 < margin_quantile < 1.0:
                threshold = float(np.quantile(np.asarray(margin_values, dtype=np.float64), margin_quantile))
            else:
                threshold = -float("inf")
            for src_u, dst_u, t_u, margin_u in update_buffer:
                if margin_u >= threshold:
                    _update_stats(stats, src_u, dst_u, t_u, max_history_per_src)
                    pseudo_edges += 1
            update_buffer.clear()
            margin_values.clear()
            last_time = time_value
        src = int(row_obj["src"])
        candidates = row_obj[cand_cols].to_numpy(np.int64)
        raw = score_sequential_candidates(src, time_value, candidates, stats, weights)
        prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
        out_rows[int(row_obj["_row_id"])] = [f"{float(x):.8f}" for x in prob]
        order = np.argsort(-raw)
        if len(order) > 1:
            margin = float(raw[order[0]] - raw[order[1]])
        else:
            margin = 0.0
        margin_values.append(margin)
        for idx in order[: max(1, int(update_topk))]:
            update_buffer.append((src, int(candidates[int(idx)]), time_value, margin))
        rows += 1
        if rows == 1 or rows % 20000 == 0:
            print({"dataset": dataset, "rows": rows, "pseudo_edges": pseudo_edges}, flush=True)
    if margin_values and 0.0 < margin_quantile < 1.0:
        threshold = float(np.quantile(np.asarray(margin_values, dtype=np.float64), margin_quantile))
    else:
        threshold = -float("inf")
    for src_u, dst_u, t_u, margin_u in update_buffer:
        if margin_u >= threshold:
            _update_stats(stats, src_u, dst_u, t_u, max_history_per_src)
            pseudo_edges += 1
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for out in out_rows:
            if out is None:
                raise RuntimeError("missing output row")
            writer.writerow(out)
    print(
        {
            "dataset": dataset,
            "history_edges": len(train),
            "candidate_pool": len(pool),
            "rows": rows,
            "pseudo_edges": pseudo_edges,
            "mode": mode,
        },
        flush=True,
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/sequential_heuristic_hard")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_online_seq_fullhistory_rank")
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "row_sigmoid", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--max-history-per-src", type=int, default=18)
    parser.add_argument("--max-pairs-per-src", type=int, default=18)
    parser.add_argument("--update-topk", type=int, default=1)
    parser.add_argument("--margin-quantile", type=float, default=0.0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset] = _write_dataset(
            dataset,
            Path(args.data_root),
            Path(args.model_dir),
            output_dir / f"{dataset}.csv",
            include_valid_history=args.include_valid_history,
            mode=args.mode,
            temperature=args.temperature,
            max_history_per_src=args.max_history_per_src,
            max_pairs_per_src=args.max_pairs_per_src,
            update_topk=args.update_topk,
            margin_quantile=args.margin_quantile,
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows})


if __name__ == "__main__":
    main()
