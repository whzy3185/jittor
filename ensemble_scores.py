from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from track1_dynamic_rec.data import find_test_candidate_file, wide_candidate_columns


def _find_col(columns, aliases):
    lower = {str(c).lower(): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def _rank01(values: np.ndarray) -> np.ndarray:
    if values.shape[1] <= 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(np.argsort(values, axis=1), axis=1)
    return order.astype(np.float32) / float(values.shape[1] - 1)


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "result_ranked.csv", "result_scores.csv"):
        target = path / name
        if target.exists():
            target.unlink()


def _write_zip(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("result.json", "result_ranked.csv", "result_scores.csv"):
            path = output_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--scores", nargs="+", required=True, help="One or more result_scores.csv files.")
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/track1/ensemble_scores")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    test_path = find_test_candidate_file(data_dir)
    if test_path is None:
        raise FileNotFoundError(f"No test.csv candidate file under {data_dir}")
    header = pd.read_csv(test_path, nrows=0)
    candidate_cols = wide_candidate_columns(header.columns)
    src_col = _find_col(header.columns, ("src", "source", "user", "user_id", "u"))
    time_col = _find_col(header.columns, ("time", "timestamp", "ts", "t"))
    if src_col is None or time_col is None or not candidate_cols:
        raise ValueError(f"{test_path} is not the official wide test format.")

    score_paths = [Path(p) for p in args.scores]
    for path in score_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    weights = args.weights or [1.0] * len(score_paths)
    if len(weights) != len(score_paths):
        raise ValueError("--weights length must match --scores length")
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-12)

    output_dir = Path(args.output_dir) / args.dataset if Path(args.output_dir).name == "ensemble_scores" else Path(args.output_dir)
    _prepare_output_dir(output_dir)
    score_cols = [f"c{i}" for i in range(1, len(candidate_cols) + 1)]
    test_iter = pd.read_csv(
        test_path,
        usecols=[src_col, time_col] + candidate_cols,
        chunksize=args.chunk_size,
        nrows=args.max_queries if args.max_queries and args.max_queries > 0 else None,
    )
    score_iters = [
        pd.read_csv(path, usecols=["query_id"] + score_cols, chunksize=args.chunk_size, nrows=args.max_queries if args.max_queries and args.max_queries > 0 else None)
        for path in score_paths
    ]

    result_json = output_dir / "result.json"
    ranked_csv = output_dir / "result_ranked.csv"
    scores_csv = output_dir / "result_scores.csv"
    total = 0
    first_json = True
    with result_json.open("w", encoding="utf-8") as jf, ranked_csv.open("w", newline="", encoding="utf-8") as rf, scores_csv.open("w", newline="", encoding="utf-8") as sf:
        jf.write("{\n")
        ranked_writer = csv.writer(rf)
        scores_writer = csv.writer(sf)
        ranked_writer.writerow(["query_id"] + [f"rank{i}" for i in range(1, len(candidate_cols) + 1)])
        scores_writer.writerow(["query_id"] + score_cols)
        for test_chunk, *score_chunks in zip(test_iter, *score_iters):
            n = len(test_chunk)
            cand_raw = test_chunk[candidate_cols].apply(pd.to_numeric, errors="raise").to_numpy(np.int64)
            blended = np.zeros((n, len(candidate_cols)), dtype=np.float32)
            qids = None
            for weight, score_chunk in zip(weights, score_chunks):
                if len(score_chunk) != n:
                    raise ValueError("Score chunk length does not match test chunk length.")
                values = score_chunk[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float32)
                blended += float(weight) * _rank01(values)
                if qids is None:
                    qids = score_chunk["query_id"].astype(str).tolist()
            if qids is None:
                raise ValueError("No score files were provided.")
            ranked_idx = np.argsort(-blended, axis=1)
            for i in range(n):
                qid = str(qids[i])
                ranked_items = [int(cand_raw[i, j]) for j in ranked_idx[i]]
                if not first_json:
                    jf.write(",\n")
                jf.write(f"  {json.dumps(qid)}: {json.dumps(ranked_items, ensure_ascii=False)}")
                first_json = False
                ranked_writer.writerow([qid] + ranked_items)
                scores_writer.writerow([qid] + [f"{float(x):.8f}" for x in blended[i]])
            total += n
            print(json.dumps({"ensembled_queries": int(total), "chunk_queries": int(n)}, ensure_ascii=False), flush=True)
        jf.write("\n}\n")
    zip_path = _write_zip(output_dir)
    print(json.dumps({"wrote_queries": total, "output_dir": str(output_dir), "zip": str(zip_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
