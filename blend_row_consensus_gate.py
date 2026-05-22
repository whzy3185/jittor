from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, values.size - 1)


def _margin(values: np.ndarray) -> float:
    idx = np.argpartition(values, -2)[-2:]
    top = np.sort(values[idx])
    return float(top[-1] - top[-2])


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


def _combine_row(rows: dict[str, np.ndarray], mode: str) -> tuple[np.ndarray, str]:
    base = rows["base"]
    replay = rows["replay"]
    more_replay = rows["more_replay"]
    diverse = rows["diverse"]
    stable = rows["stable"]
    old = rows["old"]

    tops = {k: int(np.argmax(v)) for k, v in rows.items()}
    votes = Counter([tops["base"], tops["replay"], tops["more_replay"], tops["diverse"], tops["stable"], tops["old"]])
    majority_top, majority_votes = votes.most_common(1)[0]
    base_margin = _margin(base)
    replay_margin = _margin(replay)

    if mode == "protect":
        if tops["base"] == majority_top and majority_votes >= 3:
            return 0.78 * base + 0.12 * _rank01(more_replay) + 0.10 * _rank01(diverse), "base_majority"
        if tops["more_replay"] == tops["replay"] == majority_top and majority_votes >= 3:
            return 0.58 * _rank01(more_replay) + 0.30 * base + 0.12 * _rank01(stable), "replay_majority"
        if tops["diverse"] == majority_top and majority_votes >= 3:
            return 0.46 * _rank01(diverse) + 0.40 * base + 0.14 * _rank01(stable), "diverse_majority"
        return 0.88 * base + 0.08 * _rank01(stable) + 0.04 * _rank01(old), "protect_fallback"

    if mode == "replay":
        if tops["replay"] == tops["more_replay"] and replay_margin >= 0.75 * max(base_margin, 1e-12):
            return 0.48 * _rank01(more_replay) + 0.30 * _rank01(replay) + 0.22 * base, "strong_replay"
        if tops["more_replay"] == majority_top and majority_votes >= 2:
            return 0.50 * _rank01(more_replay) + 0.32 * base + 0.18 * _rank01(diverse), "more_replay_vote"
        if tops["base"] == majority_top and majority_votes >= 3:
            return 0.70 * base + 0.18 * _rank01(more_replay) + 0.12 * _rank01(diverse), "base_majority"
        return 0.62 * base + 0.24 * _rank01(more_replay) + 0.14 * _rank01(stable), "replay_fallback"

    if mode == "diverse":
        if tops["diverse"] == majority_top and majority_votes >= 2:
            return 0.42 * _rank01(diverse) + 0.36 * base + 0.22 * _rank01(more_replay), "diverse_vote"
        if tops["base"] == tops["stable"]:
            return 0.64 * base + 0.20 * _rank01(stable) + 0.16 * _rank01(diverse), "stable_base"
        if tops["more_replay"] == tops["replay"]:
            return 0.36 * _rank01(more_replay) + 0.34 * base + 0.30 * _rank01(diverse), "replay_diverse"
        return 0.74 * base + 0.16 * _rank01(diverse) + 0.10 * _rank01(old), "diverse_fallback"

    raise ValueError(mode)


def _write_dataset2(
    dirs: dict[str, Path],
    output_csv: Path,
    chunksize: int,
    mode: str,
) -> tuple[int, dict[str, float]]:
    keys = ["base", "replay", "more_replay", "diverse", "stable", "old"]
    iters = [pd.read_csv(dirs[k] / "dataset2.csv", header=None, chunksize=chunksize) for k in keys]
    rows = 0
    changed_top1 = 0
    buckets: Counter[str] = Counter()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunks in zip(*iters):
            mats = {k: chunk.to_numpy(np.float64) for k, chunk in zip(keys, chunks)}
            n = len(chunks[0])
            out = np.empty_like(mats["base"])
            for i in range(n):
                row_dict = {k: mats[k][i] for k in keys}
                row, bucket = _combine_row(row_dict, mode)
                out[i] = np.clip(row, 0.0, 1.0)
                changed_top1 += int(np.argmax(out[i]) != np.argmax(row_dict["base"]))
                buckets[bucket] += 1
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += n
    stats = {"rows": float(rows), "top1_change_vs_base": float(changed_top1 / max(rows, 1))}
    stats.update({f"bucket_{k}": float(v) for k, v in sorted(buckets.items())})
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--more-replay-dir", required=True)
    parser.add_argument("--diverse-dir", required=True)
    parser.add_argument("--stable-dir", required=True)
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["protect", "replay", "diverse"], required=True)
    parser.add_argument("--chunksize", type=int, default=4096)
    args = parser.parse_args()

    dirs = {
        "base": Path(args.base_dir),
        "replay": Path(args.replay_dir),
        "more_replay": Path(args.more_replay_dir),
        "diverse": Path(args.diverse_dir),
        "stable": Path(args.stable_dir),
        "old": Path(args.old_dir),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows1 = _copy_dataset1(dirs["base"], output_dir, args.chunksize)
    rows2, stats = _write_dataset2(dirs, output_dir / "dataset2.csv", args.chunksize, args.mode)
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": {"dataset1": rows1, "dataset2": rows2}, "mode": args.mode, "stats": stats})


if __name__ == "__main__":
    main()
