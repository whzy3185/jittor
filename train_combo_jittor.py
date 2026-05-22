from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt
from jittor import nn, optim

from make_assoc_result_zip import assoc_component_matrix, build_assoc_stats
from make_temporal_result_zip import _build_stats as build_temporal_stats
from make_temporal_result_zip import _parse_csv_floats, temporal_component_matrix
from sequential_heuristic import build_sequential_stats, sequential_component_matrix
from simple_heuristic import history_only
from track1_dynamic_rec.train_utils import batch_indices, save_json, set_seed
from tune_sequential_hard import _make_candidates, _test_candidate_sources


class ComboLinearRanker(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def execute(self, x):
        return self.linear(x).view(-1)


def _split_train_val(train: pd.DataFrame, dataset: str, val_ratio: float, max_val_events: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "dataset2" and "split" in train.columns:
        hist = history_only(train, include_valid_history=False)
        val = train[train["split"].astype(int) == 1].copy()
    else:
        train = train.sort_values("time").reset_index(drop=True)
        cut = max(1, int(len(train) * (1.0 - val_ratio)))
        hist = train.iloc[:cut].copy()
        val = train.iloc[cut:].copy()
    if max_val_events > 0 and len(val) > max_val_events:
        val = val.sort_values("time").tail(max_val_events).copy()
    return hist[["src", "dst", "time"]].copy(), val[["src", "dst", "time"]].copy()


def _feature_frame(
    candidates: pd.DataFrame,
    seq_stats,
    temporal_stats,
    assoc_stats,
    windows: list[float],
    taus: list[float],
    decay_cap: int,
    assoc_history_score: int,
) -> np.ndarray:
    width = 13 + 6 + len(windows) * 2 + len(taus) + 8
    out = np.zeros((len(candidates), width), dtype=np.float32)
    for _, idx in candidates.groupby("query_id", sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        g = candidates.iloc[idx]
        src = int(g["src"].iloc[0])
        t = float(g["time"].iloc[0])
        dst = g["dst"].to_numpy(np.int64)
        sfeat = sequential_component_matrix(src, t, dst, seq_stats)
        tfeat = temporal_component_matrix(src, t, dst, temporal_stats, windows, taus, decay_cap)
        afeat = assoc_component_matrix(src, dst, assoc_stats, assoc_history_score)
        out[idx] = np.concatenate([sfeat, tfeat, afeat], axis=1).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset2", choices=["dataset1", "dataset2"])
    parser.add_argument("--output-dir", type=str, default="outputs/combo_jittor")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-val-events", type=int, default=5000)
    parser.add_argument("--negatives", type=int, default=99)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--seq-history", type=int, default=18)
    parser.add_argument("--seq-pairs", type=int, default=18)
    parser.add_argument("--assoc-context", type=int, default=8)
    parser.add_argument("--assoc-history", type=int, default=24)
    parser.add_argument("--assoc-score-history", type=int, default=16)
    parser.add_argument("--assoc-min-count", type=float, default=1.5)
    parser.add_argument("--windows", type=str, default="604800,2592000,7776000,31536000")
    parser.add_argument("--taus", type=str, default="604800,2592000,7776000")
    parser.add_argument("--decay-cap", type=int, default=256)
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    set_seed(args.seed)
    data_dir = Path(args.data_root) / args.dataset
    train = pd.read_csv(data_dir / "train.csv")
    hist, val = _split_train_val(train, args.dataset, args.val_ratio, args.max_val_events)
    test_pool, by_src = _test_candidate_sources(data_dir / "test.csv")
    candidates = _make_candidates(hist, val, test_pool, by_src, args.negatives, args.seed)
    candidate_pool = set(int(x) for x in test_pool)
    windows = _parse_csv_floats(args.windows)
    taus = _parse_csv_floats(args.taus)
    seq_stats = build_sequential_stats(hist, candidate_pool=candidate_pool, max_history_per_src=args.seq_history, max_pairs_per_src=args.seq_pairs)
    temporal_stats = build_temporal_stats(hist, candidate_pool=candidate_pool, keep_pair_times=True)
    assoc_stats = build_assoc_stats(
        hist,
        candidate_pool=candidate_pool,
        max_context=args.assoc_context,
        max_history_per_src=args.assoc_history,
        min_count=args.assoc_min_count,
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "history_edges": len(hist),
                "val_edges": len(val),
                "candidates": len(candidates),
                "transitions": len(seq_stats.trans_count),
                "assoc_pairs": len(assoc_stats.assoc),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    feat = _feature_frame(candidates, seq_stats, temporal_stats, assoc_stats, windows, taus, args.decay_cap, args.assoc_score_history)
    labels = candidates["label"].to_numpy(np.float32)
    qids = candidates["query_id"].to_numpy()
    pos_idx = np.where(labels > 0.5)[0]
    neg_mat = []
    for qid in candidates.iloc[pos_idx]["query_id"]:
        idx = np.where(qids == qid)[0]
        neg_mat.append(idx[labels[idx] < 0.5])
    neg_idx = np.asarray(neg_mat, dtype=np.int64)
    pos_feat = feat[pos_idx]
    neg_feat = feat[neg_idx]
    all_feat = np.concatenate([pos_feat, neg_feat.reshape(-1, feat.shape[1])], axis=0)
    mean = all_feat.mean(axis=0).astype(np.float32)
    std = all_feat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    pos_feat = ((pos_feat - mean) / std).astype(np.float32)
    neg_feat = ((neg_feat - mean) / std).astype(np.float32)
    print(json.dumps({"feature_matrix": list(feat.shape), "train_queries": int(len(pos_feat))}, ensure_ascii=False), flush=True)

    model = ComboLinearRanker(pos_feat.shape[1])
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(pos_feat)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for idx in batch_indices(n, args.batch_size, shuffle=True, seed=args.seed + epoch):
            p = model(jt.array(pos_feat[idx]))
            nf = neg_feat[idx].reshape((-1, neg_feat.shape[-1]))
            ns = model(jt.array(nf)).reshape((len(idx), neg_feat.shape[1]))
            loss = -jt.log(jt.sigmoid(p.unsqueeze(1) - ns) + 1e-8).mean()
            opt.step(loss)
            losses.append(float(loss.item()))
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses))}, ensure_ascii=False), flush=True)

    weights = model.linear.weight.detach().numpy().reshape(-1).astype(float)
    bias = float(model.linear.bias.detach().numpy().reshape(-1)[0])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = (
        [f"seq_{i}" for i in range(13)]
        + [f"temporal_{i}" for i in range(6 + len(windows) * 2 + len(taus))]
        + [f"assoc_{i}" for i in range(8)]
    )
    save_json(
        output_dir / f"{args.dataset}_combo_jittor.json",
        {
            "dataset": args.dataset,
            "feature_names": feature_names,
            "weights": weights.tolist(),
            "bias": bias,
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "windows": windows,
            "taus": taus,
            "args": vars(args),
        },
    )
    print(json.dumps({"wrote": str(output_dir / f"{args.dataset}_combo_jittor.json"), "weights_abs_mean": float(np.abs(weights).mean())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
