from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


COMPONENT_NAMES = [
    "pair_log",
    "pop_log",
    "pair_recency",
    "dst_recency",
    "src_recency",
    "pair_share",
    "candidate_prior",
    "src_degree_log",
]

DEFAULT_WEIGHTS = np.asarray([3.4, 1.15, 2.2, 0.45, 0.25, 0.8, 0.15, 0.0], dtype=np.float64)


@dataclass
class SimpleStats:
    dst_pop: Dict[int, int]
    src_degree: Dict[int, int]
    pair_count: Dict[Tuple[int, int], int]
    last_pair: Dict[Tuple[int, int], float]
    last_dst: Dict[int, float]
    last_src: Dict[int, float]


class OnlineSimpleStats:
    def __init__(self):
        self.dst_pop = Counter()
        self.src_degree = Counter()
        self.pair_count = Counter()
        self.last_pair: Dict[Tuple[int, int], float] = {}
        self.last_dst: Dict[int, float] = {}
        self.last_src: Dict[int, float] = {}

    def add_edge(self, src: int, dst: int, t: float) -> None:
        src = int(src)
        dst = int(dst)
        key = (src, dst)
        self.dst_pop[dst] += 1
        self.src_degree[src] += 1
        self.pair_count[key] += 1
        self.last_pair[key] = float(t)
        self.last_dst[dst] = float(t)
        self.last_src[src] = float(t)

    def snapshot(self) -> SimpleStats:
        return SimpleStats(
            dict(self.dst_pop),
            dict(self.src_degree),
            dict(self.pair_count),
            dict(self.last_pair),
            dict(self.last_dst),
            dict(self.last_src),
        )


def history_only(train: pd.DataFrame, include_valid_history: bool = False) -> pd.DataFrame:
    if include_valid_history or "split" not in train.columns:
        return train
    split = train["split"]
    if split.dtype.kind in {"i", "u", "f"}:
        return train[split.astype(int) == 0].copy()
    normalized = split.astype(str).str.lower().str.strip()
    return train[normalized.isin(["0", "train", "training"])].copy()


def build_simple_stats(train: pd.DataFrame) -> SimpleStats:
    src_col, dst_col = "src", "dst"
    dst_pop = train[dst_col].value_counts().to_dict()
    src_degree = train[src_col].value_counts().to_dict()
    pair_count = train.groupby([src_col, dst_col], sort=False).size().to_dict()
    if "time" in train.columns:
        last_pair = train.groupby([src_col, dst_col], sort=False)["time"].max().to_dict()
        last_dst = train.groupby(dst_col, sort=False)["time"].max().to_dict()
        last_src = train.groupby(src_col, sort=False)["time"].max().to_dict()
    else:
        last_pair, last_dst, last_src = {}, {}, {}
    return SimpleStats(dst_pop, src_degree, pair_count, last_pair, last_dst, last_src)


def component_matrix(src: int, time_value: float, candidates: np.ndarray, stats: SimpleStats) -> np.ndarray:
    out = np.zeros((len(candidates), len(COMPONENT_NAMES)), dtype=np.float64)
    src_deg = stats.src_degree.get(int(src), 0)
    src_last = stats.last_src.get(int(src), None)
    src_recency = 0.0 if src_last is None else 1.0 / math.log1p(max(1.0, time_value - float(src_last)))
    for i, dst in enumerate(candidates):
        dst = int(dst)
        pair = stats.pair_count.get((int(src), dst), 0)
        pop = stats.dst_pop.get(dst, 0)
        lp = stats.last_pair.get((int(src), dst), None)
        ld = stats.last_dst.get(dst, None)
        pair_recency = 0.0 if lp is None else 1.0 / math.log1p(max(1.0, time_value - float(lp)))
        dst_recency = 0.0 if ld is None else 1.0 / math.log1p(max(1.0, time_value - float(ld)))
        pair_share = pair / max(1, src_deg)
        candidate_prior = 1.0 / math.log2(i + 2.0)
        out[i] = [
            math.log1p(pair),
            math.log1p(pop),
            pair_recency,
            dst_recency,
            src_recency,
            pair_share,
            candidate_prior,
            math.log1p(src_deg),
        ]
    return out


def score_candidates(src: int, time_value: float, candidates: np.ndarray, stats: SimpleStats, weights: np.ndarray) -> np.ndarray:
    return component_matrix(src, time_value, candidates, stats) @ weights
