from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from track1_dynamic_rec.data import find_test_candidate_file, wide_candidate_columns


def _load_result(path: Path):
    if path.is_dir():
        result_path = path / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            with zf.open("result.json") as f:
                return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--submission", type=str, required=True, help="result.json, output directory, or zip containing result.json.")
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    test_path = find_test_candidate_file(data_dir)
    if test_path is None:
        raise FileNotFoundError(f"No official test candidate file under {data_dir}")
    header = pd.read_csv(test_path, nrows=0)
    candidate_cols = wide_candidate_columns(header.columns)
    if not candidate_cols:
        raise ValueError(f"{test_path} has no c1..cK candidate columns")

    result = _load_result(Path(args.submission))
    expected_rows = pd.read_csv(test_path, usecols=candidate_cols, nrows=args.max_queries if args.max_queries and args.max_queries > 0 else None)
    if len(result) != len(expected_rows):
        raise ValueError(f"query count mismatch: result={len(result)} expected={len(expected_rows)}")

    bad = []
    for i, (_, row) in enumerate(expected_rows.iterrows()):
        key = str(i)
        if key not in result:
            bad.append(f"missing key {key}")
            break
        pred = result[key]
        if not isinstance(pred, list):
            bad.append(f"key {key} is not a list")
            break
        expected = [int(x) for x in row.to_numpy()]
        if len(pred) != len(expected):
            bad.append(f"key {key} length {len(pred)} expected {len(expected)}")
            break
        pred_int = [int(x) for x in pred]
        if sorted(pred_int) != sorted(expected):
            bad.append(f"key {key} candidate set does not match official test row")
            break
    if bad:
        raise ValueError("; ".join(bad))
    print(
        json.dumps(
            {
                "submission": args.submission,
                "dataset": args.dataset,
                "queries": len(expected_rows),
                "candidates_per_query": len(candidate_cols),
                "valid": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
