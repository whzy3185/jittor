from __future__ import annotations

import argparse
import bisect
import csv
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from simple_heuristic import history_only
from track1_dynamic_rec.data import wide_candidate_columns


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


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


class TemporalStats:
    def __init__(self):
        self.dst_times: dict[int, np.ndarray] = {}
        self.pair_times: dict[tuple[int, int], np.ndarray] = {}
        self.src_times: dict[int, np.ndarray] = {}
        self.last_dst: dict[int, float] = {}
        self.last_pair: dict[tuple[int, int], float] = {}
        self.last_src: dict[int, float] = {}
        self.dst_count: dict[int, int] = {}
        self.pair_count: dict[tuple[int, int], int] = {}


def _build_stats(train: pd.DataFrame, candidate_pool: set[int] | None, keep_pair_times: bool) -> TemporalStats:
    train = train.sort_values("time").reset_index(drop=True)
    dst_times: dict[int, list[float]] = defaultdict(list)
    pair_times: dict[tuple[int, int], list[float]] = defaultdict(list)
    src_times: dict[int, list[float]] = defaultdict(list)
    for row in train.itertuples(index=False):
        src = int(row.src)
        dst = int(row.dst)
        t = float(row.time)
        if candidate_pool is not None and dst not in candidate_pool:
            continue
        dst_times[dst].append(t)
        src_times[src].append(t)
        if keep_pair_times:
            pair_times[(src, dst)].append(t)
    stats = TemporalStats()
    stats.dst_times = {k: np.asarray(v, dtype=np.float64) for k, v in dst_times.items()}
    stats.src_times = {k: np.asarray(v, dtype=np.float64) for k, v in src_times.items()}
    stats.pair_times = {k: np.asarray(v, dtype=np.float64) for k, v in pair_times.items()}
    stats.last_dst = {k: float(v[-1]) for k, v in stats.dst_times.items() if len(v)}
    stats.last_src = {k: float(v[-1]) for k, v in stats.src_times.items() if len(v)}
    stats.last_pair = {k: float(v[-1]) for k, v in stats.pair_times.items() if len(v)}
    stats.dst_count = {k: int(len(v)) for k, v in stats.dst_times.items()}
    stats.pair_count = {k: int(len(v)) for k, v in stats.pair_times.items()}
    return stats


def _count_since(times: np.ndarray | None, t: float, window: float) -> int:
    if times is None or len(times) == 0:
        return 0
    left = bisect.bisect_left(times, t - window)
    right = bisect.bisect_left(times, t)
    return max(0, right - left)


def _decay_sum(times: np.ndarray | None, t: float, tau: float, cap: int) -> float:
    if times is None or len(times) == 0:
        return 0.0
    left = max(0, bisect.bisect_left(times, t - tau * 8.0))
    right = bisect.bisect_left(times, t)
    if right <= left:
        return 0.0
    recent = times[max(left, right - cap) : right]
    delta = np.maximum(0.0, t - recent)
    return float(np.exp(-delta / max(tau, 1e-9)).sum())


def _score_row(src: int, t: float, candidates: np.ndarray, stats: TemporalStats, windows: list[float], taus: list[float], weights: np.ndarray, decay_cap: int) -> np.ndarray:
    return temporal_component_matrix(src, t, candidates, stats, windows, taus, decay_cap) @ weights


def temporal_component_matrix(src: int, t: float, candidates: np.ndarray, stats: TemporalStats, windows: list[float], taus: list[float], decay_cap: int) -> np.ndarray:
    out = np.zeros((len(candidates), 6 + len(windows) * 2 + len(taus)), dtype=np.float64)
    raw = np.zeros(len(candidates), dtype=np.float64)
    src_last = stats.last_src.get(int(src))
    src_recency = 0.0 if src_last is None else 1.0 / math.log1p(max(1.0, t - float(src_last)))
    for i, dst_raw in enumerate(candidates):
        dst = int(dst_raw)
        dtimes = stats.dst_times.get(dst)
        ptimes = stats.pair_times.get((int(src), dst))
        pair_total = int(stats.pair_count.get((int(src), dst), 0))
        dst_total = int(stats.dst_count.get(dst, 0))
        dst_last = stats.last_dst.get(dst)
        pair_last = stats.last_pair.get((int(src), dst))
        values: list[float] = [
            math.log1p(pair_total),
            math.log1p(dst_total),
            0.0 if pair_last is None else 1.0 / math.log1p(max(1.0, t - float(pair_last))),
            0.0 if dst_last is None else 1.0 / math.log1p(max(1.0, t - float(dst_last))),
            src_recency,
            1.0 / math.log2(i + 2.0),
        ]
        for w in windows:
            values.append(math.log1p(_count_since(dtimes, t, w)))
        for w in windows:
            values.append(math.log1p(_count_since(ptimes, t, w)))
        for tau in taus:
            values.append(math.log1p(_decay_sum(dtimes, t, tau, decay_cap)))
        out[i] = np.asarray(values, dtype=np.float64)
    return out


def _write_dataset(
    dataset: str,
    data_root: Path,
    output_csv: Path,
    include_valid_history: bool,
    chunk_size: int,
    mode: str,
    temperature: float,
    windows: list[float],
    taus: list[float],
    weights: np.ndarray,
    decay_cap: int,
) -> int:
    data_dir = data_root / dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=include_valid_history)
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    pool = _candidate_pool(test_path, cand_cols)
    stats = _build_stats(train, candidate_pool=pool, keep_pair_times=True)
    print(
        {
            "dataset": dataset,
            "history_edges": len(train),
            "dst_series": len(stats.dst_times),
            "pair_series": len(stats.pair_times),
            "windows": windows,
            "taus": taus,
        },
        flush=True,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            srcs = chunk["src"].to_numpy(np.int64)
            times = chunk["time"].to_numpy(np.float64)
            mat = chunk[cand_cols].to_numpy(np.int64)
            for src, t, row in zip(srcs, times, mat):
                raw = _score_row(int(src), float(t), row, stats, windows, taus, weights, decay_cap)
                prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_temporal_fullhistory_rank")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "minmax", "row_sigmoid"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--windows", type=str, default="604800,2592000,7776000,31536000")
    parser.add_argument("--taus", type=str, default="604800,2592000,7776000")
    parser.add_argument("--decay-cap", type=int, default=256)
    parser.add_argument(
        "--weights",
        type=str,
        default="4.0,0.05,1.2,5.0,0.5,0.0,0.8,1.0,1.2,1.0,1.2,1.4,1.4,1.0,1.2,1.0,0.8",
        help="Weights for base6 + dst window counts + pair window counts + dst decay sums.",
    )
    args = parser.parse_args()
    windows = _parse_csv_floats(args.windows)
    taus = _parse_csv_floats(args.taus)
    weights = np.asarray(_parse_csv_floats(args.weights), dtype=np.float64)
    expected = 6 + len(windows) * 2 + len(taus)
    if len(weights) != expected:
        raise ValueError(f"Expected {expected} weights, got {len(weights)}")
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
            windows=windows,
            taus=taus,
            weights=weights,
            decay_cap=args.decay_cap,
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "mode": args.mode, "weights": weights.tolist()})


if __name__ == "__main__":
    main()
