from __future__ import annotations

import argparse
import csv
import math
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from track1_dynamic_rec.data import wide_candidate_columns


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, values.size - 1)


def _prob(values: np.ndarray, mode: str) -> np.ndarray:
    values = values.astype(np.float64)
    if mode == "rank":
        return _rank01(values)
    if mode == "minmax":
        lo = float(values.min())
        hi = float(values.max())
        return (values - lo) / max(hi - lo, 1e-12)
    if mode == "sigmoid":
        z = (values - float(values.mean())) / max(float(values.std()), 1e-6)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(mode)


def _load_teacher_top1(teacher_dirs: list[Path], dataset: str) -> list[np.ndarray]:
    out = []
    for d in teacher_dirs:
        mat = pd.read_csv(d / f"{dataset}.csv", header=None).to_numpy(np.float64)
        out.append(np.argmax(mat, axis=1).astype(np.int16))
    return out


def _collect_stats(test_path: Path, cand_cols: list[str], chunk_size: int, recent_limit: int):
    global_count: Counter[int] = Counter()
    src_count: dict[int, Counter[int]] = defaultdict(Counter)
    src_col_sum: dict[int, Counter[int]] = defaultdict(Counter)
    first_col_sum: Counter[int] = Counter()
    row_count_by_src: Counter[int] = Counter()
    recent_by_src: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=recent_limit))

    for chunk in pd.read_csv(test_path, chunksize=chunk_size):
        srcs = chunk["src"].to_numpy(np.int64)
        mat = chunk[cand_cols].to_numpy(np.int64)
        for src_raw, row in zip(srcs, mat):
            src = int(src_raw)
            row_count_by_src[src] += 1
            seen_in_row: set[int] = set()
            for j, dst_raw in enumerate(row):
                dst = int(dst_raw)
                global_count[dst] += 1
                src_count[src][dst] += 1
                src_col_sum[src][dst] += j + 1
                if dst not in seen_in_row:
                    first_col_sum[dst] += j + 1
                    seen_in_row.add(dst)
    return global_count, src_count, src_col_sum, first_col_sum, row_count_by_src, recent_by_src


def _write_dataset(
    dataset: str,
    data_root: Path,
    teacher_dirs: list[Path],
    output_csv: Path,
    chunk_size: int,
    mode: str,
    recent_limit: int,
    w_srcfreq: float,
    w_globalfreq: float,
    w_mean_col: float,
    w_col_prior: float,
    w_recent: float,
    w_teacher_top: float,
    w_row_dup: float,
) -> tuple[int, dict[str, float]]:
    test_path = data_root / dataset / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    teachers = _load_teacher_top1(teacher_dirs, dataset) if teacher_dirs else []
    global_count, src_count, src_col_sum, first_col_sum, row_count_by_src, recent_by_src = _collect_stats(
        test_path, cand_cols, chunk_size, recent_limit
    )
    rows = 0
    teacher_hits = 0
    duplicate_rows = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            srcs = chunk["src"].to_numpy(np.int64)
            mat = chunk[cand_cols].to_numpy(np.int64)
            for local_i, (src_raw, row) in enumerate(zip(srcs, mat)):
                row_id = rows + local_i
                src = int(src_raw)
                raw = np.zeros(len(row), dtype=np.float64)
                row_seen: Counter[int] = Counter(int(x) for x in row)
                duplicate_rows += int(any(v > 1 for v in row_seen.values()))
                teacher_vote: Counter[int] = Counter()
                for top1 in teachers:
                    teacher_vote[int(top1[row_id])] += 1
                recent = recent_by_src[src]
                recent_counter = Counter(recent)
                src_bucket = src_count.get(src, {})
                col_bucket = src_col_sum.get(src, {})
                for j, dst_raw in enumerate(row):
                    dst = int(dst_raw)
                    sf = int(src_bucket.get(dst, 0))
                    gf = int(global_count.get(dst, 0))
                    mean_col = float(col_bucket.get(dst, j + 1)) / max(1, sf)
                    raw[j] = (
                        w_srcfreq * math.log1p(sf)
                        + w_globalfreq * math.log1p(gf)
                        + w_mean_col / math.log2(mean_col + 1.0)
                        + w_col_prior / math.log2(j + 2.0)
                        + w_recent * math.log1p(recent_counter.get(dst, 0))
                        + w_teacher_top * teacher_vote.get(j, 0)
                        + w_row_dup * max(0, row_seen.get(dst, 1) - 1)
                    )
                prob = np.clip(_prob(raw, mode), 0.0, 1.0)
                writer.writerow([f"{float(x):.8f}" for x in prob])

                top_idx = int(np.argmax(prob))
                teacher_hits += int(teacher_vote.get(top_idx, 0) > 0)
                # Online update uses only previous candidate rows, not labels.
                recent.append(int(row[top_idx]))
            rows += len(chunk)
            if rows == len(chunk) or rows % (chunk_size * 10) == 0:
                print({"dataset": dataset, "rows": rows}, flush=True)
    return rows, {
        "rows": float(rows),
        "unique_global_dst": float(len(global_count)),
        "unique_src": float(len(src_count)),
        "teacher_hit_rate": float(teacher_hits / max(rows, 1)),
        "duplicate_row_rate": float(duplicate_rows / max(rows, 1)),
        "src_rows_p50": float(np.quantile(np.asarray(list(row_count_by_src.values()), dtype=np.float64), 0.5)),
        "src_rows_p90": float(np.quantile(np.asarray(list(row_count_by_src.values()), dtype=np.float64), 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--teacher-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--mode", choices=["rank", "minmax", "sigmoid"], default="rank")
    parser.add_argument("--recent-limit", type=int, default=20)
    parser.add_argument("--w-srcfreq", type=float, default=2.5)
    parser.add_argument("--w-globalfreq", type=float, default=0.25)
    parser.add_argument("--w-mean-col", type=float, default=0.6)
    parser.add_argument("--w-col-prior", type=float, default=0.1)
    parser.add_argument("--w-recent", type=float, default=1.2)
    parser.add_argument("--w-teacher-top", type=float, default=1.0)
    parser.add_argument("--w-row-dup", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    teacher_dirs = [Path(x) for x in args.teacher_dirs]
    rows = {}
    stats = {}
    for dataset in ("dataset1", "dataset2"):
        rows[dataset], stats[dataset] = _write_dataset(
            dataset,
            Path(args.data_root),
            teacher_dirs,
            output_dir / f"{dataset}.csv",
            args.chunk_size,
            args.mode,
            args.recent_limit,
            args.w_srcfreq,
            args.w_globalfreq,
            args.w_mean_col,
            args.w_col_prior,
            args.w_recent,
            args.w_teacher_top,
            args.w_row_dup,
        )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "stats": stats})


if __name__ == "__main__":
    main()
