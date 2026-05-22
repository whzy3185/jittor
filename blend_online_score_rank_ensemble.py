from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return order.astype(np.float64) / max(1, values.size - 1)


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


def _write_dataset2(
    base_dir: Path,
    expert_dirs: list[Path],
    weights: np.ndarray,
    output_csv: Path,
    chunksize: int,
    preserve_base_scale: float,
) -> tuple[int, dict[str, float]]:
    paths = [d / "dataset2.csv" for d in expert_dirs]
    iters = [pd.read_csv(path, header=None, chunksize=chunksize) for path in paths]
    base_iter = pd.read_csv(base_dir / "dataset2.csv", header=None, chunksize=chunksize)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    changed_top1 = 0
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for chunks in zip(base_iter, *iters):
            base = chunks[0].to_numpy(np.float64)
            mats = [chunk.to_numpy(np.float64) for chunk in chunks[1:]]
            out = np.empty_like(base)
            for i in range(len(base)):
                rank_mix = np.zeros(base.shape[1], dtype=np.float64)
                for w, mat in zip(weights, mats):
                    rank_mix += float(w) * _rank01(mat[i])
                if preserve_base_scale > 0:
                    sorted_scores = np.sort(base[i])[::-1]
                    order = np.argsort(-rank_mix)
                    row = np.empty_like(rank_mix)
                    row[order] = sorted_scores
                    row = (1.0 - preserve_base_scale) * rank_mix + preserve_base_scale * row
                else:
                    row = rank_mix
                out[i] = np.clip(row, 0.0, 1.0)
                changed_top1 += int(np.argmax(out[i]) != np.argmax(base[i]))
            for row in out:
                writer.writerow([f"{float(x):.8f}" for x in row])
            rows += len(base)
    return rows, {"top1_change_vs_base": float(changed_top1 / max(rows, 1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--expert-dirs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preserve-base-scale", type=float, default=0.0)
    parser.add_argument("--chunksize", type=int, default=4096)
    args = parser.parse_args()
    if len(args.expert_dirs) != len(args.weights):
        raise ValueError("--expert-dirs and --weights length mismatch")
    weights = np.asarray(args.weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    output_dir = Path(args.output_dir)
    base_dir = Path(args.base_dir)
    rows1 = _copy_dataset1(base_dir, output_dir, args.chunksize)
    rows2, stats = _write_dataset2(
        base_dir,
        [Path(x) for x in args.expert_dirs],
        weights,
        output_dir / "dataset2.csv",
        args.chunksize,
        float(args.preserve_base_scale),
    )
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print(
        {
            "zip": str(zip_path),
            "rows": {"dataset1": rows1, "dataset2": rows2},
            "weights": weights.tolist(),
            "preserve_base_scale": float(args.preserve_base_scale),
            "stats": stats,
        }
    )


if __name__ == "__main__":
    main()
