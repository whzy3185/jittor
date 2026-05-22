from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from enhanced_heuristic import (
    ENHANCED_COMPONENT_NAMES,
    ENHANCED_DEFAULT_WEIGHTS,
    EnhancedStats,
    build_enhanced_stats,
    enhanced_component_matrix,
)


SEQUENTIAL_COMPONENT_NAMES = ENHANCED_COMPONENT_NAMES + [
    "trans_log",
    "trans_max",
    "trans_last",
]

SEQUENTIAL_DEFAULT_WEIGHTS = np.concatenate(
    [
        ENHANCED_DEFAULT_WEIGHTS,
        np.asarray([1.8, 1.2, 1.5], dtype=np.float64),
    ]
)


@dataclass
class SequentialStats:
    enhanced: EnhancedStats
    trans_count: Dict[Tuple[int, int], int]
    trans_out: Dict[int, int]


def build_sequential_stats(
    train: pd.DataFrame,
    candidate_pool: set[int] | None = None,
    max_history_per_src: int = 30,
    max_pairs_per_src: int = 30,
) -> SequentialStats:
    train = train.sort_values("time").reset_index(drop=True) if "time" in train.columns else train.reset_index(drop=True)
    enhanced = build_enhanced_stats(
        train,
        candidate_pool=candidate_pool,
        max_history_per_src=max_history_per_src,
        max_pairs_per_src=max_pairs_per_src,
    )
    trans_count: Counter[tuple[int, int]] = Counter()
    trans_out: Counter[int] = Counter()
    for _, group in train.groupby("src", sort=False):
        seq = group.sort_values("time")["dst"].to_numpy(np.int64)
        if len(seq) < 2:
            continue
        for prev, nxt in zip(seq[:-1], seq[1:]):
            prev = int(prev)
            nxt = int(nxt)
            if prev == nxt:
                continue
            if candidate_pool is not None and nxt not in candidate_pool:
                continue
            trans_count[(prev, nxt)] += 1
            trans_out[prev] += 1
    return SequentialStats(enhanced=enhanced, trans_count=dict(trans_count), trans_out=dict(trans_out))


def transition_features(src: int, candidates: np.ndarray, stats: SequentialStats, max_history_score: int = 12) -> np.ndarray:
    out = np.zeros((len(candidates), 3), dtype=np.float64)
    hist = stats.enhanced.recent_by_src.get(int(src))
    if hist is None or len(hist) == 0:
        return out
    hist = hist[:max_history_score]
    dst_pop = stats.enhanced.simple.dst_pop
    for i, dst in enumerate(candidates):
        dst = int(dst)
        pd = max(1, int(dst_pop.get(dst, 0)))
        total = 0.0
        best = 0.0
        last_score = 0.0
        for rank, prev in enumerate(hist):
            prev = int(prev)
            c = stats.trans_count.get((prev, dst), 0)
            if c <= 0:
                continue
            out_degree = max(1, int(stats.trans_out.get(prev, 0)))
            v = c / math.sqrt(float(out_degree * pd))
            v = v / math.log2(rank + 2.0)
            total += v
            best = max(best, v)
            if rank == 0:
                last_score = v
        out[i, 0] = math.log1p(total)
        out[i, 1] = best
        out[i, 2] = last_score
    return out


def sequential_component_matrix(src: int, time_value: float, candidates: np.ndarray, stats: SequentialStats) -> np.ndarray:
    base = enhanced_component_matrix(src, time_value, candidates, stats.enhanced)
    extra = transition_features(src, candidates, stats)
    return np.concatenate([base, extra], axis=1)


def score_sequential_candidates(src: int, time_value: float, candidates: np.ndarray, stats: SequentialStats, weights: np.ndarray) -> np.ndarray:
    return sequential_component_matrix(src, time_value, candidates, stats) @ weights
