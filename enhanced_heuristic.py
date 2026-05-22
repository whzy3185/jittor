from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from simple_heuristic import COMPONENT_NAMES as SIMPLE_COMPONENT_NAMES
from simple_heuristic import DEFAULT_WEIGHTS as SIMPLE_DEFAULT_WEIGHTS
from simple_heuristic import SimpleStats, build_simple_stats, component_matrix, history_only


ENHANCED_COMPONENT_NAMES = SIMPLE_COMPONENT_NAMES + [
    "cooc_log",
    "cooc_max",
]

ENHANCED_DEFAULT_WEIGHTS = np.concatenate(
    [
        SIMPLE_DEFAULT_WEIGHTS,
        np.asarray([1.2, 0.8], dtype=np.float64),
    ]
)


@dataclass
class EnhancedStats:
    simple: SimpleStats
    recent_by_src: Dict[int, np.ndarray]
    cooc: Dict[Tuple[int, int], int]


def _pair_key(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _recent_unique(values: Iterable[int], limit: int) -> np.ndarray:
    seen: set[int] = set()
    out: list[int] = []
    for value in reversed(list(values)):
        x = int(value)
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= limit:
            break
    return np.asarray(out, dtype=np.int64)


def build_enhanced_stats(
    train: pd.DataFrame,
    candidate_pool: set[int] | None = None,
    max_history_per_src: int = 30,
    max_pairs_per_src: int = 30,
) -> EnhancedStats:
    train = train.sort_values("time").reset_index(drop=True) if "time" in train.columns else train.reset_index(drop=True)
    simple = build_simple_stats(train)
    recent_by_src: dict[int, np.ndarray] = {}
    cooc_counter: Counter[tuple[int, int]] = Counter()
    src_groups = train.groupby("src", sort=False)["dst"]
    for src, series in src_groups:
        recent = _recent_unique(series.to_numpy(np.int64), max_history_per_src)
        recent_by_src[int(src)] = recent
        pair_items = recent[:max_pairs_per_src]
        n = len(pair_items)
        for i in range(n):
            a = int(pair_items[i])
            for j in range(i + 1, n):
                b = int(pair_items[j])
                if candidate_pool is not None and a not in candidate_pool and b not in candidate_pool:
                    continue
                cooc_counter[_pair_key(a, b)] += 1
    return EnhancedStats(simple=simple, recent_by_src=recent_by_src, cooc=dict(cooc_counter))


def cooc_features(src: int, candidates: np.ndarray, stats: EnhancedStats, max_history_score: int = 12) -> np.ndarray:
    out = np.zeros((len(candidates), 2), dtype=np.float64)
    hist = stats.recent_by_src.get(int(src))
    if hist is None or len(hist) == 0:
        return out
    hist = hist[:max_history_score]
    dst_pop = stats.simple.dst_pop
    for i, dst in enumerate(candidates):
        dst = int(dst)
        pd = max(1, int(dst_pop.get(dst, 0)))
        total = 0.0
        best = 0.0
        for h in hist:
            h = int(h)
            if h == dst:
                continue
            c = stats.cooc.get(_pair_key(dst, h), 0)
            if c <= 0:
                continue
            ph = max(1, int(dst_pop.get(h, 0)))
            v = c / math.sqrt(float(pd * ph))
            total += v
            if v > best:
                best = v
        out[i, 0] = math.log1p(total)
        out[i, 1] = best
    return out


def enhanced_component_matrix(src: int, time_value: float, candidates: np.ndarray, stats: EnhancedStats) -> np.ndarray:
    base = component_matrix(src, time_value, candidates, stats.simple)
    extra = cooc_features(src, candidates, stats)
    return np.concatenate([base, extra], axis=1)


def score_enhanced_candidates(src: int, time_value: float, candidates: np.ndarray, stats: EnhancedStats, weights: np.ndarray) -> np.ndarray:
    feat = enhanced_component_matrix(src, time_value, candidates, stats)
    return feat @ weights
