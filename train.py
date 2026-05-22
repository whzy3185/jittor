from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt
from jittor import nn
from jittor import optim

from track1_dynamic_rec.data import describe_bundle, load_competition_data
from track1_dynamic_rec.features import build_candidate_features, build_training_pairs, fit_feature_stats, resolve_time_window
from track1_dynamic_rec.metrics import rank_normalize, ranking_metrics
from track1_dynamic_rec.model import HybridTemporalScorer
from track1_dynamic_rec.train_utils import batch_indices, save_json, set_seed


def make_optimizer(model, lr: float, weight_decay: float):
    if hasattr(optim, "AdamW"):
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _select_blend(out: pd.DataFrame, default_model_weight: float, default_heuristic_weight: float) -> tuple[pd.DataFrame, dict]:
    if "label" not in out.columns:
        out["score"] = default_model_weight * out["model_rank"] + default_heuristic_weight * out["heuristic_rank"]
        return out, ranking_metrics(out)

    best_metrics = None
    best_weight = float(default_model_weight)
    for weight in np.linspace(0.0, 1.0, 21):
        tmp = out.copy()
        tmp["score"] = float(weight) * tmp["model_rank"] + float(1.0 - weight) * tmp["heuristic_rank"]
        metrics = ranking_metrics(tmp)
        key = metrics.get("mrr", metrics.get("auc", -1e9))
        best_key = best_metrics.get("mrr", best_metrics.get("auc", -1e9)) if best_metrics else -1e9
        if key > best_key:
            best_metrics = metrics
            best_weight = float(weight)
    out["score"] = best_weight * out["model_rank"] + (1.0 - best_weight) * out["heuristic_rank"]
    metrics = dict(best_metrics or ranking_metrics(out))
    metrics["blend_model_weight"] = best_weight
    metrics["blend_heuristic_weight"] = float(1.0 - best_weight)
    return out, metrics


def evaluate(model, candidates: pd.DataFrame, history: pd.DataFrame, stats, batch_size: int, window: float, model_weight: float, heuristic_weight: float):
    if candidates is None or len(candidates) == 0:
        return {}, None
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
    out, metrics = _select_blend(out, model_weight, heuristic_weight)
    return metrics, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None, help="Overrides --data-root/--dataset when set.")
    parser.add_argument("--output-dir", type=str, default="outputs/track1")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Temporal holdout ratio for datasets without an official split. Use 0 for final all-history training.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--val-negatives", type=int, default=50)
    parser.add_argument("--max-val-events", type=int, default=0, help="Limit validation positive events before generating local negatives.")
    parser.add_argument("--max-test-queries", type=int, default=0, help="Limit test queries for smoke-test ID mapping. Use 0 for all official test IDs.")
    parser.add_argument("--load-test-candidates", action="store_true", help="Expand official test candidates during training. Usually unnecessary and memory-heavy.")
    parser.add_argument("--window", type=float, default=0.0, help="Recent-history window in native time unit. 0 enables auto scaling.")
    parser.add_argument("--model-weight", type=float, default=0.78)
    parser.add_argument("--heuristic-weight", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    set_seed(args.seed)
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    output_dir = Path(args.output_dir) / args.dataset if Path(args.output_dir).name == "track1" else Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_competition_data(
        data_dir,
        val_ratio=args.val_ratio,
        num_val_negatives=args.val_negatives,
        seed=args.seed,
        max_val_events=args.max_val_events,
        max_test_queries=args.max_test_queries,
        load_test_candidates=args.load_test_candidates,
        map_test_candidates=True,
    )
    print(describe_bundle(bundle))
    window = resolve_time_window(bundle.train_edges, args.window)
    time_shift = float(bundle.train_edges["time"].min()) if len(bundle.train_edges) else 0.0
    time_scale = max(1.0, window)
    print(json.dumps({"dataset": args.dataset, "data_dir": str(data_dir), "window": window, "time_shift": time_shift, "time_scale": time_scale}, ensure_ascii=False))
    pairs = build_training_pairs(
        bundle.train_edges,
        num_negatives=args.negatives,
        seed=args.seed,
        window=window,
        max_events=args.max_train_events if args.max_train_events > 0 else None,
    )
    if len(pairs["src"]) == 0:
        raise RuntimeError("No training pairs were produced. Check train data and negative sampling.")
    train_feat_all = np.concatenate([pairs["pos_feat"], pairs["neg_feat"].reshape(-1, pairs["neg_feat"].shape[-1])], axis=0)
    stats = fit_feature_stats(train_feat_all)
    pairs["pos_feat"] = stats.transform(pairs["pos_feat"]).astype(np.float32)
    pairs["neg_feat"] = stats.transform(pairs["neg_feat"]).astype(np.float32)

    model = HybridTemporalScorer(
        num_nodes=bundle.num_nodes,
        num_features=pairs["pos_feat"].shape[1],
        emb_dim=args.emb_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        time_shift=time_shift,
        time_scale=time_scale,
    )
    opt = make_optimizer(model, args.lr, args.weight_decay)
    best_metric = -1e9
    bad = 0
    history_log = []

    n = len(pairs["src"])
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx in batch_indices(n, args.batch_size, shuffle=True, seed=args.seed + epoch):
            loss, _, _ = model.pairwise_loss(
                jt.array(pairs["src"][idx]),
                jt.array(pairs["pos_dst"][idx]),
                jt.array(pairs["neg_dst"][idx]),
                jt.array(pairs["time"][idx]),
                jt.array(pairs["pos_feat"][idx]),
                jt.array(pairs["neg_feat"][idx]),
            )
            opt.step(loss)
            losses.append(float(loss.item()))
        metrics, _ = evaluate(model, bundle.val_candidates, bundle.train_edges, stats, args.batch_size, window, args.model_weight, args.heuristic_weight)
        key_metric = metrics.get("mrr", metrics.get("auc", -np.mean(losses)))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history_log.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if key_metric > best_metric:
            best_metric = key_metric
            bad = 0
            jt.save(model.state_dict(), str(ckpt_dir / "best.pkl"))
            id_mapping_path = ckpt_dir / "id_mapping.json"
            save_json(id_mapping_path, {str(k): int(v) for k, v in sorted(bundle.id_mapping.items())})
            save_json(
                ckpt_dir / "best_meta.json",
                {
                    "args": vars(args),
                    "feature_stats": stats.to_dict(),
                    "num_nodes": bundle.num_nodes,
                    "num_features": int(pairs["pos_feat"].shape[1]),
                    "best_metric": float(best_metric),
                    "bundle_notes": bundle.notes,
                    "data_dir": str(data_dir),
                    "dataset": args.dataset,
                    "window": float(window),
                    "time_shift": float(time_shift),
                    "time_scale": float(time_scale),
                    "max_test_queries": int(args.max_test_queries),
                    "id_mapping_path": str(id_mapping_path),
                    "blend_model_weight": float(metrics.get("blend_model_weight", args.model_weight)),
                    "blend_heuristic_weight": float(metrics.get("blend_heuristic_weight", args.heuristic_weight)),
                },
            )
        else:
            bad += 1
            if bad >= args.patience:
                break
    pd.DataFrame(history_log).to_csv(output_dir / "train_log.csv", index=False)
    print(f"best_metric={best_metric:.6f} checkpoint={ckpt_dir / 'best.pkl'}")


if __name__ == "__main__":
    main()
