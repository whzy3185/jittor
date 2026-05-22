from __future__ import annotations

import argparse
import csv
import math
import zipfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from simple_heuristic import history_only
from track1_dynamic_rec.data import wide_candidate_columns


SHIFT = 32
MASK = (1 << SHIFT) - 1


def _pack(a: int, b: int) -> int:
    return (int(a) << SHIFT) | int(b)


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, len(values) - 1)


def _prob(values: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    values = values.astype(np.float64)
    if mode == "rank":
        return _rank01(values)
    if mode == "minmax":
        lo = float(values.min())
        hi = float(values.max())
        return (values - lo) / max(hi - lo, 1e-12)
    if mode == "row_sigmoid":
        z = (values - float(values.mean())) / max(float(values.std()) * temperature, 1e-6)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(mode)


def _candidate_pool(test_path: Path, cand_cols: list[str]) -> set[int]:
    out: set[int] = set()
    for chunk in pd.read_csv(test_path, usecols=cand_cols, chunksize=8192):
        out.update(int(x) for x in chunk.to_numpy(np.int64).reshape(-1))
    return out


@dataclass
class AssocStats:
    assoc: dict[int, float]
    assoc_out: dict[int, float]
    dst_pop: dict[int, int]
    pair_count: dict[tuple[int, int], int]
    recent_by_src: dict[int, np.ndarray]


def _push_unique(hist: deque[int], dst: int, limit: int) -> None:
    dst = int(dst)
    try:
        hist.remove(dst)
    except ValueError:
        pass
    hist.appendleft(dst)
    while len(hist) > limit:
        hist.pop()


def build_assoc_stats(
    train: pd.DataFrame,
    candidate_pool: set[int] | None,
    max_context: int,
    max_history_per_src: int,
    min_count: float,
) -> AssocStats:
    train = train.sort_values(["src", "time"]).reset_index(drop=True)
    assoc_counter: Counter[int] = Counter()
    assoc_out: Counter[int] = Counter()
    recent_by_src: dict[int, np.ndarray] = {}
    pair_count: Counter[tuple[int, int]] = Counter()
    dst_pop: Counter[int] = Counter()
    for src, group in train.groupby("src", sort=False):
        hist: deque[int] = deque()
        src_i = int(src)
        for row in group.itertuples(index=False):
            dst = int(row.dst)
            dst_pop[dst] += 1
            pair_count[(src_i, dst)] += 1
            if candidate_pool is None or dst in candidate_pool:
                for rank, prev in enumerate(list(hist)[:max_context]):
                    if prev == dst:
                        continue
                    w = 1.0 / math.log2(rank + 2.0)
                    key = _pack(prev, dst)
                    assoc_counter[key] += w
                    assoc_out[prev] += w
            _push_unique(hist, dst, max(max_context, max_history_per_src))
        recent_by_src[src_i] = np.asarray(list(hist)[:max_history_per_src], dtype=np.int64)
    if min_count > 0:
        assoc = {int(k): float(v) for k, v in assoc_counter.items() if float(v) >= min_count}
    else:
        assoc = {int(k): float(v) for k, v in assoc_counter.items()}
    return AssocStats(
        assoc=assoc,
        assoc_out={int(k): float(v) for k, v in assoc_out.items()},
        dst_pop={int(k): int(v) for k, v in dst_pop.items()},
        pair_count=dict(pair_count),
        recent_by_src=recent_by_src,
    )


def assoc_component_matrix(src: int, candidates: np.ndarray, stats: AssocStats, max_history_score: int) -> np.ndarray:
    out = np.zeros((len(candidates), 8), dtype=np.float64)
    hist = stats.recent_by_src.get(int(src))
    if hist is None or len(hist) == 0:
        return out
    hist = hist[:max_history_score]
    for i, dst_raw in enumerate(candidates):
        dst = int(dst_raw)
        dst_pop = max(1, int(stats.dst_pop.get(dst, 0)))
        total = 0.0
        best = 0.0
        last = 0.0
        hit_count = 0.0
        pmi_total = 0.0
        for rank, prev_raw in enumerate(hist):
            prev = int(prev_raw)
            v = float(stats.assoc.get(_pack(prev, dst), 0.0))
            if v <= 0:
                continue
            hit_count += 1.0
            prev_out = max(1.0, float(stats.assoc_out.get(prev, 0.0)))
            norm = v / math.sqrt(prev_out * float(dst_pop))
            norm = norm / math.log2(rank + 2.0)
            pmi = math.log1p(v * 1000.0 / math.sqrt(prev_out * float(dst_pop)))
            total += norm
            pmi_total += pmi / math.log2(rank + 2.0)
            best = max(best, norm)
            if rank == 0:
                last = norm
        pair = int(stats.pair_count.get((int(src), dst), 0))
        out[i] = [
            math.log1p(total),
            best,
            last,
            math.log1p(hit_count),
            math.log1p(pmi_total),
            math.log1p(pair),
            pair / max(1.0, float(len(hist))),
            1.0 / math.log2(i + 2.0),
        ]
    return out


def score_assoc_candidates(src: int, candidates: np.ndarray, stats: AssocStats, weights: np.ndarray, max_history_score: int) -> np.ndarray:
    return assoc_component_matrix(src, candidates, stats, max_history_score) @ weights


def _write_dataset(
    dataset: str,
    data_root: Path,
    output_csv: Path,
    include_valid_history: bool,
    chunk_size: int,
    mode: str,
    temperature: float,
    max_context: int,
    max_history_per_src: int,
    max_history_score: int,
    min_count: float,
    weights: np.ndarray,
) -> int:
    data_dir = data_root / dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=include_valid_history)
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    pool = _candidate_pool(test_path, cand_cols)
    stats = build_assoc_stats(
        train,
        candidate_pool=pool,
        max_context=max_context,
        max_history_per_src=max_history_per_src,
        min_count=min_count,
    )
    print(
        {
            "dataset": dataset,
            "history_edges": len(train),
            "candidate_pool": len(pool),
            "assoc_pairs": len(stats.assoc),
            "max_context": max_context,
            "min_count": min_count,
        },
        flush=True,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            srcs = chunk["src"].to_numpy(np.int64)
            mat = chunk[cand_cols].to_numpy(np.int64)
            for src, row in zip(srcs, mat):
                raw = score_assoc_candidates(int(src), row, stats, weights, max_history_score)
                prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_assoc_fullhistory_rank")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "minmax", "row_sigmoid"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--max-context", type=int, default=8)
    parser.add_argument("--max-history-per-src", type=int, default=24)
    parser.add_argument("--max-history-score", type=int, default=16)
    parser.add_argument("--min-count", type=float, default=1.0)
    parser.add_argument("--weights", type=str, default="2.0,1.5,1.2,0.4,1.1,2.5,1.0,0.0")
    args = parser.parse_args()
    weights = np.asarray([float(x) for x in args.weights.split(",") if x.strip()], dtype=np.float64)
    if len(weights) != 8:
        raise ValueError("--weights must contain 8 numbers")
    output_dir = Path(args.output_dir)
    rows = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset] = _write_dataset(
            dataset,
            Path(args.data_root),
            output_dir / f"{dataset}.csv",
            include_valid_history=args.include_valid_history,
            chunk_size=args.chunk_size,
            mode=args.mode,
            temperature=args.temperature,
            max_context=args.max_context,
            max_history_per_src=args.max_history_per_src,
            max_history_score=args.max_history_score,
            min_count=args.min_count,
            weights=weights,
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode, "weights": weights.tolist()})


if __name__ == "__main__":
    main()
