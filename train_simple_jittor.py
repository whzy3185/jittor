from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt
from jittor import nn, optim

from simple_heuristic import COMPONENT_NAMES, OnlineSimpleStats, component_matrix, history_only
from track1_dynamic_rec.train_utils import batch_indices, save_json, set_seed


PRIOR_IDX = COMPONENT_NAMES.index("candidate_prior")


def _candidate_cols(columns) -> list[str]:
    cols = [c for c in columns if str(c).lower().startswith("c") and str(c)[1:].isdigit()]
    return sorted(cols, key=lambda x: int(str(x)[1:]))


def _test_pool(data_dir: Path) -> np.ndarray:
    test_path = data_dir / "test.csv"
    cols = _candidate_cols(pd.read_csv(test_path, nrows=0).columns)
    values = []
    for chunk in pd.read_csv(test_path, usecols=cols, chunksize=8192):
        values.append(chunk.to_numpy(np.int64).reshape(-1))
    return np.unique(np.concatenate(values).astype(np.int64))


def _build_pair_features(
    train: pd.DataFrame,
    test_pool: np.ndarray,
    max_events: int,
    negatives: int,
    seed: int,
    warmup_events: int,
    use_candidate_prior: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train = train.sort_values("time").reset_index(drop=True)
    if max_events > 0 and len(train) > max_events + warmup_events:
        start = len(train) - max_events - warmup_events
        train = train.iloc[start:].reset_index(drop=True)
    dst_values, dst_counts = np.unique(train["dst"].to_numpy(np.int64), return_counts=True)
    pop_prob = np.power(dst_counts.astype(np.float64), 0.75)
    pop_prob = pop_prob / pop_prob.sum()
    positive_by_src: dict[int, set[int]] = {}
    for s, d in zip(train["src"].to_numpy(np.int64), train["dst"].to_numpy(np.int64)):
        positive_by_src.setdefault(int(s), set()).add(int(d))
    online = OnlineSimpleStats()
    pos_feats = []
    neg_feats = []
    src_arr = train["src"].to_numpy(np.int64)
    dst_arr = train["dst"].to_numpy(np.int64)
    time_arr = train["time"].to_numpy(np.float64)
    for i in range(len(train)):
        src, dst, t = int(src_arr[i]), int(dst_arr[i]), float(time_arr[i])
        if i >= warmup_events:
            stats = online
            pf = component_matrix(src, t, np.asarray([dst], dtype=np.int64), stats)[0]
            if not use_candidate_prior:
                pf[PRIOR_IDX] = 0.0
            ndsts = []
            tries = 0
            positives = positive_by_src.get(src, set())
            while len(ndsts) < negatives and tries < negatives * 100 + 200:
                tries += 1
                if rng.rand() < 0.65 and len(test_pool):
                    ndst = int(test_pool[rng.randint(0, len(test_pool))])
                else:
                    ndst = int(rng.choice(dst_values, p=pop_prob))
                if ndst == dst or ndst in ndsts or ndst in positives:
                    continue
                ndsts.append(ndst)
            if len(ndsts) == negatives:
                nf = component_matrix(src, t, np.asarray(ndsts, dtype=np.int64), stats)
                if not use_candidate_prior:
                    nf[:, PRIOR_IDX] = 0.0
                pos_feats.append(pf)
                neg_feats.append(nf)
        online.add_edge(src, dst, t)
    if not pos_feats:
        raise RuntimeError("No training feature pairs were produced.")
    return np.asarray(pos_feats, dtype=np.float32), np.asarray(neg_feats, dtype=np.float32)


class LinearRanker(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def execute(self, x):
        return self.linear(x).view(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--output-dir", type=str, default="outputs/simple_jittor")
    parser.add_argument("--max-events", type=int, default=200000)
    parser.add_argument("--warmup-events", type=int, default=10000)
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-candidate-prior", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    set_seed(args.seed)
    data_dir = Path(args.data_root) / args.dataset
    train = pd.read_csv(data_dir / "train.csv")
    train = history_only(train, include_valid_history=False)
    pool = _test_pool(data_dir)
    pos, neg = _build_pair_features(
        train,
        pool,
        args.max_events,
        args.negatives,
        args.seed,
        args.warmup_events,
        args.use_candidate_prior,
    )
    all_feat = np.concatenate([pos, neg.reshape(-1, neg.shape[-1])], axis=0)
    mean = all_feat.mean(axis=0).astype(np.float32)
    std = all_feat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    pos = ((pos - mean) / std).astype(np.float32)
    neg = ((neg - mean) / std).astype(np.float32)

    model = LinearRanker(pos.shape[1])
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    w = model.linear.weight.detach().numpy().reshape(-1).astype(float)
    b = float(model.linear.bias.detach().numpy().reshape(-1)[0])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / f"{args.dataset}_simple_jittor.json",
        {
            "dataset": args.dataset,
            "component_names": COMPONENT_NAMES,
            "weights": w.tolist(),
            "bias": b,
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "args": vars(args),
            "use_candidate_prior": bool(args.use_candidate_prior),
            "num_pairs": int(n),
        },
    )
    print(json.dumps({"wrote": str(output_dir / f"{args.dataset}_simple_jittor.json"), "num_pairs": int(n)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
