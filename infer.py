from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import jittor as jt

from track1_dynamic_rec.data import describe_bundle, load_competition_data
from track1_dynamic_rec.features import FeatureStats, build_candidate_features, resolve_time_window
from track1_dynamic_rec.metrics import rank_normalize, ranking_metrics
from track1_dynamic_rec.model import HybridTemporalScorer
from track1_dynamic_rec.submission import package_submission, write_submission
from track1_dynamic_rec.train_utils import batch_indices, load_json


def score_candidates(model, candidates, history, stats, batch_size: int, window: float, model_weight: float, heuristic_weight: float):
    feat, heur = build_candidate_features(candidates, history, window=window)
    feat = stats.transform(feat).astype(np.float32)
    scores = np.zeros(len(candidates), dtype=np.float32)
    model.eval()
    for idx in batch_indices(len(candidates), batch_size, shuffle=False):
        src = jt.array(candidates.iloc[idx]["src"].to_numpy(np.int64))
        dst = jt.array(candidates.iloc[idx]["dst"].to_numpy(np.int64))
        t = jt.array(candidates.iloc[idx]["time"].to_numpy(np.float32))
        f = jt.array(feat[idx])
        scores[idx] = model.score(src, dst, t, f).detach().numpy()
    out = candidates.copy()
    out["model_score"] = scores
    out["heuristic_score"] = heur
    out["model_rank"] = rank_normalize(out.assign(tmp_score=scores), "tmp_score")
    out["heuristic_rank"] = rank_normalize(out.assign(tmp_score=heur), "tmp_score")
    out["score"] = model_weight * out["model_rank"] + heuristic_weight * out["heuristic_rank"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None, help="Overrides --data-root/--dataset when set.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/track1/submission")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-val-events", type=int, default=0, help="Limit validation positives when reporting local validation during inference.")
    parser.add_argument("--max-test-queries", type=int, default=0, help="Limit loaded official test queries for smoke tests. Must match smoke checkpoint mapping.")
    parser.add_argument("--window", type=float, default=None)
    parser.add_argument("--model-weight", type=float, default=None, help="Override checkpoint blend_model_weight.")
    parser.add_argument("--heuristic-weight", type=float, default=None, help="Override checkpoint blend_heuristic_weight.")
    parser.add_argument("--include-valid-in-test-history", action="store_true")
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
        num_val_negatives=int(train_args.get("val_negatives", 50)),
        seed=int(train_args.get("seed", 42)),
        max_val_events=args.max_val_events,
        max_test_queries=args.max_test_queries or int(meta.get("max_test_queries", 0)),
    )
    print(describe_bundle(bundle))
    window = args.window if args.window is not None else float(meta.get("window", resolve_time_window(bundle.train_edges, 0.0)))
    time_shift = float(meta.get("time_shift", bundle.train_edges["time"].min() if len(bundle.train_edges) else 0.0))
    time_scale = float(meta.get("time_scale", max(1.0, window)))
    model_weight = float(args.model_weight) if args.model_weight is not None else float(meta.get("blend_model_weight", 0.82))
    heuristic_weight = float(args.heuristic_weight) if args.heuristic_weight is not None else float(meta.get("blend_heuristic_weight", 1.0 - model_weight))

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

    if bundle.val_candidates is not None and len(bundle.val_candidates):
        val_pred = score_candidates(model, bundle.val_candidates, bundle.train_edges, stats, args.batch_size, window, model_weight, heuristic_weight)
        print("validation", ranking_metrics(val_pred))

    if bundle.test_candidates is None or len(bundle.test_candidates) == 0:
        raise RuntimeError("No official test candidates were found. Place the official test candidate file under --data-dir.")
    test_history = bundle.all_edges if args.include_valid_in_test_history else bundle.train_edges
    pred = score_candidates(model, bundle.test_candidates, test_history, stats, args.batch_size, window, model_weight, heuristic_weight)
    output_dir = Path(args.output_dir) / args.dataset if Path(args.output_dir).name == "submission" else Path(args.output_dir)
    result_path = write_submission(pred, output_dir, bundle.sample_submission)
    zip_path = package_submission(output_dir, output_dir.with_suffix(".zip"))
    print(f"wrote {result_path}")
    print(f"packaged {zip_path}")


if __name__ == "__main__":
    main()
