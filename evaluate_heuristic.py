from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from track1_dynamic_rec.data import describe_bundle, load_competition_data
from track1_dynamic_rec.features import build_candidate_features, resolve_time_window
from track1_dynamic_rec.metrics import rank_normalize, ranking_metrics
from track1_dynamic_rec.train_utils import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/track1/heuristic_eval")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-negatives", type=int, default=50)
    parser.add_argument("--max-val-events", type=int, default=5000)
    parser.add_argument("--max-test-queries", type=int, default=0)
    parser.add_argument("--window", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    output_dir = Path(args.output_dir) / args.dataset if Path(args.output_dir).name == "heuristic_eval" else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_competition_data(
        data_dir,
        val_ratio=args.val_ratio,
        num_val_negatives=args.val_negatives,
        seed=args.seed,
        max_val_events=args.max_val_events,
        max_test_queries=args.max_test_queries,
        load_test_candidates=False,
        map_test_candidates=True,
    )
    print(describe_bundle(bundle))
    if bundle.val_candidates is None or len(bundle.val_candidates) == 0:
        raise RuntimeError("No validation candidates available.")
    window = resolve_time_window(bundle.train_edges, args.window)
    _, heur = build_candidate_features(bundle.val_candidates, bundle.train_edges, window=window)
    pred = bundle.val_candidates.copy()
    pred["heuristic_score"] = heur
    pred["score"] = rank_normalize(pred.assign(tmp_score=heur), "tmp_score")
    metrics = ranking_metrics(pred)
    result = {
        "dataset": args.dataset,
        "data_dir": str(data_dir),
        "window": float(window),
        "metrics": metrics,
        "notes": bundle.notes,
    }
    save_json(output_dir / "metrics.json", result)
    keep = [c for c in ("query_id", "candidate_rank", "src_raw", "dst_raw", "time", "label", "heuristic_score", "score") if c in pred.columns]
    pred[keep].to_csv(output_dir / "validation_predictions.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
