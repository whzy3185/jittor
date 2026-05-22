from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import jittor as jt

from infer import score_candidates
from track1_dynamic_rec.data import describe_bundle, load_competition_data
from track1_dynamic_rec.features import FeatureStats, resolve_time_window
from track1_dynamic_rec.metrics import ranking_metrics
from track1_dynamic_rec.model import HybridTemporalScorer
from track1_dynamic_rec.train_utils import load_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-val-events", type=int, default=5000)
    parser.add_argument("--max-test-queries", type=int, default=0)
    parser.add_argument("--weight-step", type=float, default=0.025)
    parser.add_argument("--write-meta", action="store_true", help="Update best_meta.json with tuned blend weights.")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    default_run_dir = Path("outputs/track1") / args.dataset
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_run_dir / "checkpoints" / "best.pkl"
    meta_path = Path(args.meta) if args.meta else default_run_dir / "checkpoints" / "best_meta.json"
    meta = load_json(meta_path)
    train_args = meta["args"]
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    bundle = load_competition_data(
        data_dir,
        val_ratio=float(train_args.get("val_ratio", 0.1)),
        num_val_negatives=int(train_args.get("val_negatives", 50)),
        seed=int(train_args.get("seed", 42)),
        max_val_events=args.max_val_events,
        max_test_queries=args.max_test_queries,
        load_test_candidates=False,
        map_test_candidates=True,
    )
    print(describe_bundle(bundle))
    if bundle.val_candidates is None or len(bundle.val_candidates) == 0:
        raise RuntimeError("No validation candidates available. Use a run with validation holdout/split.")

    window = float(meta.get("window", resolve_time_window(bundle.train_edges, 0.0)))
    time_shift = float(meta.get("time_shift", bundle.train_edges["time"].min() if len(bundle.train_edges) else 0.0))
    time_scale = float(meta.get("time_scale", max(1.0, window)))
    stats = FeatureStats.from_dict(meta["feature_stats"])
    model = HybridTemporalScorer(
        num_nodes=int(meta["num_nodes"]),
        num_features=int(meta["num_features"]),
        emb_dim=int(train_args.get("emb_dim", 128)),
        time_dim=int(train_args.get("time_dim", 32)),
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        num_layers=int(train_args.get("layers", 3)),
        dropout=float(train_args.get("dropout", 0.15)),
        time_shift=time_shift,
        time_scale=time_scale,
    )
    model.load_state_dict(jt.load(str(checkpoint)))
    pred = score_candidates(model, bundle.val_candidates, bundle.train_edges, stats, args.batch_size, window, 1.0, 0.0)

    best = None
    step = max(1e-6, float(args.weight_step))
    weights = np.arange(0.0, 1.0 + step * 0.5, step)
    rows = []
    for weight in weights:
        pred["score"] = float(weight) * pred["model_rank"] + float(1.0 - weight) * pred["heuristic_rank"]
        metrics = ranking_metrics(pred)
        row = {
            "model_weight": float(weight),
            "heuristic_weight": float(1.0 - weight),
            **metrics,
        }
        rows.append(row)
        key = metrics.get("mrr", metrics.get("auc", -1e9))
        best_key = best.get("mrr", best.get("auc", -1e9)) if best else -1e9
        if key > best_key:
            best = row
    print(json.dumps({"best": best, "checkpoint": str(checkpoint), "meta": str(meta_path)}, ensure_ascii=False, indent=2))
    out = {"best": best, "grid": rows}
    output_json = Path(args.output_json) if args.output_json else meta_path.parent / "blend_tuning.json"
    save_json(output_json, out)
    print(f"wrote {output_json}")
    if args.write_meta:
        meta["blend_model_weight"] = float(best["model_weight"])
        meta["blend_heuristic_weight"] = float(best["heuristic_weight"])
        meta["blend_tuning_path"] = str(output_json)
        save_json(meta_path, meta)
        print(f"updated {meta_path}")


if __name__ == "__main__":
    main()
