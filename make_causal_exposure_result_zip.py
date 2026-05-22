from __future__ import annotations

import argparse
import csv
import math
import shutil
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

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


class ExposureStats:
    def __init__(self, recent_limit: int):
        self.global_count: Counter[int] = Counter()
        self.global_col_sum: Counter[int] = Counter()
        self.global_last_time: dict[int, float] = {}
        self.src_count: dict[int, Counter[int]] = defaultdict(Counter)
        self.src_col_sum: dict[int, Counter[int]] = defaultdict(Counter)
        self.src_last_time: dict[tuple[int, int], float] = {}
        self.src_seen_rows: Counter[int] = Counter()
        self.src_recent: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=recent_limit))
        self.recent_limit = int(recent_limit)

    def update_group(self, group: pd.DataFrame, cand_cols: list[str]) -> None:
        srcs = group["src"].to_numpy(np.int64)
        times = group["time"].to_numpy(np.float64)
        mat = group[cand_cols].to_numpy(np.int64)
        for src_raw, t_raw, row in zip(srcs, times, mat):
            src = int(src_raw)
            t = float(t_raw)
            self.src_seen_rows[src] += 1
            sbucket = self.src_count[src]
            scol = self.src_col_sum[src]
            recent = self.src_recent[src]
            for j, dst_raw in enumerate(row):
                dst = int(dst_raw)
                col = j + 1
                self.global_count[dst] += 1
                self.global_col_sum[dst] += col
                self.global_last_time[dst] = t
                sbucket[dst] += 1
                scol[dst] += col
                self.src_last_time[(src, dst)] = t
                recent.appendleft(dst)


def _score_row(src: int, t: float, row: np.ndarray, stats: ExposureStats, weights: np.ndarray) -> np.ndarray:
    raw = np.zeros(len(row), dtype=np.float64)
    src_bucket = stats.src_count.get(src, {})
    src_col = stats.src_col_sum.get(src, {})
    recent = stats.src_recent.get(src)
    recent_rank: dict[int, int] = {}
    if recent:
        for rank, dst in enumerate(recent):
            if int(dst) not in recent_rank:
                recent_rank[int(dst)] = rank + 1
    src_rows = int(stats.src_seen_rows.get(src, 0))
    for j, dst_raw in enumerate(row):
        dst = int(dst_raw)
        gc = int(stats.global_count.get(dst, 0))
        sc = int(src_bucket.get(dst, 0))
        gmean_col = float(stats.global_col_sum.get(dst, j + 1)) / max(1, gc)
        smean_col = float(src_col.get(dst, j + 1)) / max(1, sc)
        glast = stats.global_last_time.get(dst)
        slast = stats.src_last_time.get((src, dst))
        gr = 0.0 if glast is None else 1.0 / math.log1p(max(1.0, t - float(glast)))
        sr = 0.0 if slast is None else 1.0 / math.log1p(max(1.0, t - float(slast)))
        rr = recent_rank.get(dst)
        recent_hit = 0.0 if rr is None else 1.0 / math.log2(float(rr) + 1.0)
        src_share = sc / max(1.0, float(src_rows))
        col_prior = 1.0 / math.log2(j + 2.0)
        features = np.asarray(
            [
                math.log1p(sc),
                src_share,
                sr,
                1.0 / math.log2(smean_col + 1.0) if sc > 0 else 0.0,
                recent_hit,
                math.log1p(gc),
                gr,
                1.0 / math.log2(gmean_col + 1.0) if gc > 0 else 0.0,
                col_prior,
            ],
            dtype=np.float64,
        )
        raw[j] = float(features @ weights)
    return raw


def _parse_weights(text: str) -> np.ndarray:
    weights = np.asarray([float(x) for x in text.split(",") if x.strip()], dtype=np.float64)
    if len(weights) != 9:
        raise ValueError("--weights must contain 9 values")
    return weights


def _write_dataset(
    data_root: Path,
    dataset: str,
    output_csv: Path,
    mode: str,
    temperature: float,
    weights: np.ndarray,
    recent_limit: int,
) -> tuple[int, dict[str, float]]:
    test_path = data_root / dataset / "test.csv"
    header = pd.read_csv(test_path, nrows=0)
    cand_cols = wide_candidate_columns(header.columns)
    test = pd.read_csv(test_path)
    test = test.assign(_row_id=test.index.to_numpy(np.int64)).sort_values(["time", "_row_id"]).reset_index(drop=True)
    out_rows: list[list[str] | None] = [None] * len(test)
    stats = ExposureStats(recent_limit=recent_limit)
    rows = 0
    groups = 0
    for time_value, group in test.groupby("time", sort=False):
        # Score the entire timestamp before updating exposure stats, preventing same-time leakage.
        srcs = group["src"].to_numpy(np.int64)
        times = group["time"].to_numpy(np.float64)
        row_ids = group["_row_id"].to_numpy(np.int64)
        mat = group[cand_cols].to_numpy(np.int64)
        for src_raw, t_raw, row_id_raw, candidates in zip(srcs, times, row_ids, mat):
            src = int(src_raw)
            t = float(t_raw)
            row_id = int(row_id_raw)
            raw = _score_row(src, t, candidates, stats, weights)
            prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
            out_rows[row_id] = [f"{float(x):.8f}" for x in prob]
            rows += 1
        stats.update_group(group, cand_cols)
        groups += 1
        if groups == 1 or rows % 20000 < len(group):
            print(
                {
                    "dataset": dataset,
                    "rows": rows,
                    "time_groups": groups,
                    "global_exposed": len(stats.global_count),
                    "src_groups": len(stats.src_seen_rows),
                },
                flush=True,
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in out_rows:
            if row is None:
                raise RuntimeError("missing output row")
            writer.writerow(row)
    return rows, {
        "time_groups": float(groups),
        "global_exposed": float(len(stats.global_count)),
        "src_groups": float(len(stats.src_seen_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--baseline-dir", type=str, default="outputs/website_submission_89_major_teacher88_online_top4q80_40")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dataset2", choices=["dataset1", "dataset2"])
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "minmax", "row_sigmoid"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--recent-limit", type=int, default=512)
    parser.add_argument(
        "--weights",
        type=str,
        default="3.5,2.0,1.6,1.0,1.4,0.35,0.5,0.25,0.05",
        help="src_log,src_share,src_recency,src_col,src_recent,global_log,global_recency,global_col,col_prior",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = Path(args.baseline_dir)
    shutil.copyfile(baseline / "dataset1.csv", output_dir / "dataset1.csv")
    if args.dataset == "dataset1":
        target_csv = output_dir / "dataset1.csv"
    else:
        target_csv = output_dir / "dataset2.csv"
    rows2, stats = _write_dataset(
        Path(args.data_root),
        args.dataset,
        target_csv,
        args.mode,
        args.temperature,
        _parse_weights(args.weights),
        args.recent_limit,
    )
    if args.dataset == "dataset1":
        shutil.copyfile(baseline / "dataset2.csv", output_dir / "dataset2.csv")
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "dataset": args.dataset, "rows": rows2, "stats": stats})


if __name__ == "__main__":
    main()
