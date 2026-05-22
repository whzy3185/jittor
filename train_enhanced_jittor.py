from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt
from jittor import nn, optim

from enhanced_heuristic import ENHANCED_COMPONENT_NAMES, build_enhanced_stats, enhanced_component_matrix
from simple_heuristic import history_only
from track1_dynamic_rec.train_utils import batch_indices, save_json, set_seed


PRIOR_IDX = ENHANCED_COMPONENT_NAMES.index("candidate_prior")


def _candidate_cols(columns) -> list[str]:
    cols = [c for c in columns if str(c).lower().startswith("c") and str(c)[1:].isdigit()]
    return sorted(cols, key=lambda x: int(str(x)[1:]))


def _test_pool(data_dir: Path) -> np.ndarray:
    cols = _candidate_cols(pd.read_csv(data_dir / "test.csv", nrows=0).columns)
    values = []
    for chunk in pd.read_csv(data_dir / "test.csv", usecols=cols, chunksize=8192):
        values.append(chunk.to_numpy(np.int64).reshape(-1))
    return np.unique(np.concatenate(values).astype(np.int64))


class EnhancedMLPRanker(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x):
        h = nn.relu(self.fc1(x))
        h = self.dropout(h)
        r = h
        h = nn.relu(self.fc2(h))
        h = self.dropout(h)
        h = h + r
        return self.fc3(h).view(-1)


def _build_train_features(
    train: pd.DataFrame,
    test_pool: np.ndarray,
    split_ratio: float,
    max_events: int,
    negatives: int,
    seed: int,
    use_candidate_prior: bool,
    max_history_per_src: int,
    max_pairs_per_src: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train = train.sort_values("time").reset_index(drop=True)
    cut = max(1, min(len(train) - 1, int(len(train) * split_ratio)))
    hist = train.iloc[:cut].copy()
    future = train.iloc[cut:].copy()
    if max_events > 0 and len(future) > max_events:
        future = future.tail(max_events).copy()
    candidate_pool = set(int(x) for x in test_pool)
    stats = build_enhanced_stats(
        hist,
        candidate_pool=candidate_pool,
        max_history_per_src=max_history_per_src,
        max_pairs_per_src=max_pairs_per_src,
    )
    dst_values, dst_counts = np.unique(hist["dst"].to_numpy(np.int64), return_counts=True)
    pop_prob = np.power(dst_counts.astype(np.float64), 0.75)
    pop_prob = pop_prob / pop_prob.sum()
    positives_by_src: dict[int, set[int]] = {}
    for s, d in zip(train["src"].to_numpy(np.int64), train["dst"].to_numpy(np.int64)):
        positives_by_src.setdefault(int(s), set()).add(int(d))
    pos_feats: list[np.ndarray] = []
    neg_feats: list[np.ndarray] = []
    for row in future.itertuples(index=False):
        src, dst, t = int(row.src), int(row.dst), float(row.time)
        blocked = set(positives_by_src.get(src, set()))
        blocked.add(dst)
        ndsts: list[int] = []
        tries = 0
        while len(ndsts) < negatives and tries < negatives * 200 + 500:
            tries += 1
            if rng.rand() < 0.70 and len(test_pool):
                ndst = int(test_pool[rng.randint(0, len(test_pool))])
            else:
                ndst = int(rng.choice(dst_values, p=pop_prob))
            if ndst in blocked or ndst in ndsts:
                continue
            ndsts.append(ndst)
        if len(ndsts) != negatives:
            continue
        candidates = np.asarray([dst] + ndsts, dtype=np.int64)
        feat = enhanced_component_matrix(src, t, candidates, stats)
        if not use_candidate_prior:
            feat[:, PRIOR_IDX] = 0.0
        pos_feats.append(feat[0])
        neg_feats.append(feat[1:])
    if not pos_feats:
        raise RuntimeError("No enhanced Jittor training pairs were produced.")
    return np.asarray(pos_feats, dtype=np.float32), np.asarray(neg_feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--output-dir", type=str, default="outputs/enhanced_jittor_mlp")
    parser.add_argument("--split-ratio", type=float, default=0.85)
    parser.add_argument("--max-events", type=int, default=120000)
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-candidate-prior", action="store_true")
    parser.add_argument("--max-history-per-src", type=int, default=30)
    parser.add_argument("--max-pairs-per-src", type=int, default=30)
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    set_seed(args.seed)
    data_dir = Path(args.data_root) / args.dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=False)
    test_pool = _test_pool(data_dir)
    pos, neg = _build_train_features(
        train,
        test_pool,
        args.split_ratio,
        args.max_events,
        args.negatives,
        args.seed,
        args.use_candidate_prior,
        args.max_history_per_src,
        args.max_pairs_per_src,
    )
    all_feat = np.concatenate([pos, neg.reshape(-1, neg.shape[-1])], axis=0)
    mean = all_feat.mean(axis=0).astype(np.float32)
    std = all_feat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    pos = ((pos - mean) / std).astype(np.float32)
    neg = ((neg - mean) / std).astype(np.float32)

    model = EnhancedMLPRanker(pos.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(pos)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for idx in batch_indices(n, args.batch_size, shuffle=True, seed=args.seed + epoch):
            p = model(jt.array(pos[idx]))
            nf = neg[idx].reshape((-1, neg.shape[-1]))
            ns = model(jt.array(nf)).reshape((len(idx), neg.shape[1]))
            loss = -jt.log(jt.sigmoid(p.unsqueeze(1) - ns) + 1e-8).mean()
            opt.step(loss)
            losses.append(float(loss.item()))
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses))}, ensure_ascii=False), flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{args.dataset}_enhanced_mlp.pkl"
    jt.save(model.state_dict(), str(ckpt_path))
    save_json(
        output_dir / f"{args.dataset}_enhanced_mlp.json",
        {
            "dataset": args.dataset,
            "component_names": ENHANCED_COMPONENT_NAMES,
            "checkpoint": ckpt_path.name,
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "args": vars(args),
            "use_candidate_prior": bool(args.use_candidate_prior),
            "num_pairs": int(n),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
        },
    )
    print(json.dumps({"wrote": str(ckpt_path), "num_pairs": int(n)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
