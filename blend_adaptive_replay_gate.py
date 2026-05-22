from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _copy_dataset1(base_dir: Path, output_dir: Path, chunksize: int) -> int:
    rows = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dataset1.csv").open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunk in pd.read_csv(base_dir / "dataset1.csv", header=None, chunksize=chunksize):
            for row in chunk.to_numpy(np.float64):
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(chunk)
    return rows


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, values.size - 1)


def _row_margin(values: np.ndarray) -> float:
    idx = np.argpartition(values, -2)[-2:]
    top = np.sort(values[idx])
    return float(top[-1] - top[-2])


def _source_features(test_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test = pd.read_csv(test_csv, usecols=["src", "time"])
    test = test.assign(_row_id=test.index.to_numpy(np.int64)).sort_values(["time", "_row_id"]).reset_index(drop=True)
    total_counter = Counter(int(x) for x in test["src"].to_numpy(np.int64))
    total = np.zeros(len(test), dtype=np.int32)
    seen_before = np.zeros(len(test), dtype=np.int32)
    time_seen_before = np.zeros(len(test), dtype=np.int32)
    seen: Counter[int] = Counter()
    time_seen: Counter[int] = Counter()
    for _, group in test.groupby("time", sort=False):
        row_ids = group["_row_id"].to_numpy(np.int64)
        srcs = group["src"].to_numpy(np.int64)
        for rid, src_raw in zip(row_ids, srcs):
            src = int(src_raw)
            total[int(rid)] = int(total_counter[src])
            seen_before[int(rid)] = int(seen[src])
            time_seen_before[int(rid)] = int(time_seen[src])
        seen.update(int(x) for x in srcs)
        time_seen.clear()
        time_seen.update(int(x) for x in srcs)
    total_values = np.asarray(list(total_counter.values()), dtype=np.float64)
    thresholds = np.quantile(total_values, [0.50, 0.75, 0.90])
    return total, seen_before, time_seen_before, thresholds.astype(np.float64)


def _expert_row(paths: list[Path], chunksize: int):
    iters = [pd.read_csv(path, header=None, chunksize=chunksize) for path in paths]
    for chunks in zip(*iters):
        yield [chunk.to_numpy(np.float64) for chunk in chunks]


def _adaptive_weight(
    *,
    base: np.ndarray,
    replay: np.ndarray,
    stable_a: np.ndarray,
    stable_b: np.ndarray,
    total_src: int,
    seen_before: int,
    thresholds: np.ndarray,
    mode: str,
) -> tuple[float, str]:
    b_top = int(np.argmax(base))
    r_top = int(np.argmax(replay))
    a_top = int(np.argmax(stable_a))
    c_top = int(np.argmax(stable_b))
    top_votes = Counter([b_top, a_top, c_top])
    majority_top, majority_votes = top_votes.most_common(1)[0]
    replay_margin = _row_margin(replay)
    base_margin = _row_margin(base)
    hot = total_src >= thresholds[2]
    warm = total_src >= thresholds[1]
    active = seen_before >= 8
    very_active = seen_before >= 24

    if mode == "bold":
        if r_top == majority_top and majority_votes >= 2 and very_active:
            return 0.72, "replay_majority_hot"
        if r_top == majority_top and majority_votes >= 2 and active:
            return 0.55, "replay_majority_active"
        if r_top != b_top and r_top in (a_top, c_top) and (hot or very_active) and replay_margin >= 0.65 * max(base_margin, 1e-12):
            return 0.48, "replay_teacher_hot_disagree"
        if r_top == b_top and (warm or active):
            return 0.32, "same_top_warm"
        if b_top == majority_top and majority_votes >= 2 and r_top != b_top:
            return 0.06, "protect_base_majority"
        if hot:
            return 0.24, "hot_fallback"
        return 0.10, "cold_fallback"

    if mode == "stable":
        if r_top == majority_top and majority_votes >= 2 and very_active:
            return 0.50, "replay_majority_hot"
        if r_top == majority_top and majority_votes >= 2 and active:
            return 0.38, "replay_majority_active"
        if r_top != b_top and r_top in (a_top, c_top) and hot:
            return 0.28, "replay_teacher_hot_disagree"
        if r_top == b_top and (warm or active):
            return 0.24, "same_top_warm"
        if b_top == majority_top and majority_votes >= 2 and r_top != b_top:
            return 0.03, "protect_base_majority"
        if hot:
            return 0.16, "hot_fallback"
        return 0.05, "cold_fallback"

    raise ValueError(mode)


def _write_dataset2(
    base_dir: Path,
    replay_dir: Path,
    stable_dirs: list[Path],
    test_csv: Path,
    output_csv: Path,
    chunksize: int,
    mode: str,
    rank_space: bool,
) -> tuple[int, dict[str, float]]:
    if len(stable_dirs) != 2:
        raise ValueError("expected exactly two stable dirs")
    total_src, seen_before, _, thresholds = _source_features(test_csv)
    paths = [
        base_dir / "dataset2.csv",
        replay_dir / "dataset2.csv",
        stable_dirs[0] / "dataset2.csv",
        stable_dirs[1] / "dataset2.csv",
    ]
    rows = 0
    changed_top1 = 0
    weight_sum = 0.0
    bucket_counts: Counter[str] = Counter()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for mats in _expert_row(paths, chunksize):
            base_mat, replay_mat, a_mat, b_mat = mats
            out = np.empty_like(base_mat)
            for i in range(len(base_mat)):
                global_i = offset + i
                weight, bucket = _adaptive_weight(
                    base=base_mat[i],
                    replay=replay_mat[i],
                    stable_a=a_mat[i],
                    stable_b=b_mat[i],
                    total_src=int(total_src[global_i]),
                    seen_before=int(seen_before[global_i]),
                    thresholds=thresholds,
                    mode=mode,
                )
                if rank_space:
                    base_row = _rank01(base_mat[i])
                    replay_row = _rank01(replay_mat[i])
                else:
                    base_row = base_mat[i]
                    replay_row = replay_mat[i]
                row = (1.0 - weight) * base_row + weight * replay_row
                out[i] = np.clip(row, 0.0, 1.0)
                changed_top1 += int(np.argmax(out[i]) != np.argmax(base_mat[i]))
                weight_sum += float(weight)
                bucket_counts[bucket] += 1
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base_mat)
            offset += len(base_mat)
    stats = {
        "rows": float(rows),
        "avg_weight": float(weight_sum / max(rows, 1)),
        "top1_change_vs_base": float(changed_top1 / max(rows, 1)),
        "threshold_p50": float(thresholds[0]),
        "threshold_p75": float(thresholds[1]),
        "threshold_p90": float(thresholds[2]),
    }
    stats.update({f"bucket_{k}": float(v) for k, v in sorted(bucket_counts.items())})
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--stable-dirs", nargs=2, required=True)
    parser.add_argument("--data-root", default="data/official_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["bold", "stable"], default="bold")
    parser.add_argument("--score-space", action="store_true")
    parser.add_argument("--chunksize", type=int, default=4096)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    replay_dir = Path(args.replay_dir)
    stable_dirs = [Path(x) for x in args.stable_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _write_dataset2(
        base_dir,
        replay_dir,
        stable_dirs,
        Path(args.data_root) / "dataset2" / "test.csv",
        output_dir / "dataset2.csv",
        args.chunksize,
        args.mode,
        rank_space=not args.score_space,
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "stats": stats})


if __name__ == "__main__":
    main()
