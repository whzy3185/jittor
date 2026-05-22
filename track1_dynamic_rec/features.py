from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "src_degree_log",
    "dst_degree_log",
    "pair_count_log",
    "src_recent_log",
    "dst_recent_log",
    "pair_recent_log",
    "dst_long_recent_log",
    "src_delta_log",
    "dst_delta_log",
    "pair_delta_log",
    "common_recent_log",
    "dst_pop_rank",
    "src_activity_rank",
    "pair_share",
    "repeat_flag",
    "new_dst_for_src_flag",
]


@dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> Dict[str, list]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, list]) -> "FeatureStats":
        return cls(mean=np.asarray(d["mean"], dtype=np.float32), std=np.asarray(d["std"], dtype=np.float32))


class CausalFeatureBuilder:
    def __init__(self, window: float = 7.0):
        self.window = float(window)
        self.long_window = float(window) * 4.0 if float(window) > 0 else 0.0
        self.src_degree = Counter()
        self.dst_degree = Counter()
        self.pair_count = Counter()
        self.src_last: Dict[int, float] = {}
        self.dst_last: Dict[int, float] = {}
        self.pair_last: Dict[Tuple[int, int], float] = {}
        self.src_recent = defaultdict(deque)
        self.dst_recent = defaultdict(deque)
        self.pair_recent = defaultdict(deque)
        self.dst_long_recent = defaultdict(deque)
        self.neighbors = defaultdict(set)
        self.max_src_degree = 1
        self.max_dst_degree = 1

    def _prune(self, q: deque, t: float) -> None:
        if self.window <= 0:
            return
        min_t = t - self.window
        while q and q[0] < min_t:
            q.popleft()

    def add_edge(self, src: int, dst: int, t: float) -> None:
        key = (src, dst)
        self.src_degree[src] += 1
        self.dst_degree[dst] += 1
        if self.src_degree[src] > self.max_src_degree:
            self.max_src_degree = self.src_degree[src]
        if self.dst_degree[dst] > self.max_dst_degree:
            self.max_dst_degree = self.dst_degree[dst]
        self.pair_count[key] += 1
        self.src_last[src] = t
        self.dst_last[dst] = t
        self.pair_last[key] = t
        self.src_recent[src].append(t)
        self.dst_recent[dst].append(t)
        self.pair_recent[key].append(t)
        self.dst_long_recent[dst].append(t)
        self.neighbors[src].add(dst)
        self.neighbors[dst].add(src)

    def score(self, src: int, dst: int, t: float) -> Tuple[np.ndarray, float]:
        key = (src, dst)
        self._prune(self.src_recent[src], t)
        self._prune(self.dst_recent[dst], t)
        self._prune(self.pair_recent[key], t)
        if self.long_window > 0:
            min_t = t - self.long_window
            q = self.dst_long_recent[dst]
            while q and q[0] < min_t:
                q.popleft()
        src_deg = self.src_degree[src]
        dst_deg = self.dst_degree[dst]
        pair_cnt = self.pair_count[key]
        src_recent = len(self.src_recent[src])
        dst_recent = len(self.dst_recent[dst])
        pair_recent = len(self.pair_recent[key])
        dst_long_recent = len(self.dst_long_recent[dst])
        src_delta = 1e6 if src not in self.src_last else max(0.0, t - self.src_last[src])
        dst_delta = 1e6 if dst not in self.dst_last else max(0.0, t - self.dst_last[dst])
        pair_delta = 1e6 if key not in self.pair_last else max(0.0, t - self.pair_last[key])
        common = len(self.neighbors[src].intersection(self.neighbors[dst]))
        dst_pop_rank = dst_deg / self.max_dst_degree
        src_activity_rank = src_deg / self.max_src_degree
        pair_share = pair_cnt / max(1, src_deg)
        repeat = 1.0 if pair_cnt > 0 else 0.0
        new_dst_for_src = 1.0 if pair_cnt == 0 else 0.0
        feat = np.asarray(
            [
                np.log1p(src_deg),
                np.log1p(dst_deg),
                np.log1p(pair_cnt),
                np.log1p(src_recent),
                np.log1p(dst_recent),
                np.log1p(pair_recent),
                np.log1p(dst_long_recent),
                np.log1p(src_delta),
                np.log1p(dst_delta),
                np.log1p(pair_delta),
                np.log1p(common),
                dst_pop_rank,
                src_activity_rank,
                pair_share,
                repeat,
                new_dst_for_src,
            ],
            dtype=np.float32,
        )
        heuristic = (
            2.2 * np.log1p(pair_cnt)
            + 0.8 * np.log1p(dst_recent)
            + 0.45 * np.log1p(dst_long_recent)
            + 0.55 * np.log1p(dst_deg)
            + 0.3 * np.log1p(common)
            + 0.25 * pair_share
            - 0.10 * np.log1p(pair_delta)
        )
        return feat, float(heuristic)


def resolve_time_window(history_edges: pd.DataFrame, requested: float = 0.0) -> float:
    """Choose a leakage-safe recent window in the native timestamp unit."""
    if requested and requested > 0:
        return float(requested)
    if len(history_edges) == 0 or "time" not in history_edges:
        return 7.0
    times = history_edges["time"].to_numpy(dtype=np.float64)
    span = float(np.nanmax(times) - np.nanmin(times))
    if span <= 0:
        return 7.0
    # Unix-like seconds: use 30 days. Otherwise keep roughly the last 3% of the
    # observed training period, bounded away from a degenerate tiny window.
    if np.nanmax(times) > 10_000_000:
        return 30.0 * 24.0 * 3600.0
    return max(7.0, span * 0.03)


def fit_feature_stats(x: np.ndarray) -> FeatureStats:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return FeatureStats(mean=mean, std=std)


def build_candidate_features(candidates: pd.DataFrame, history_edges: pd.DataFrame, window: float = 7.0) -> Tuple[np.ndarray, np.ndarray]:
    builder = CausalFeatureBuilder(window=window)
    candidates = candidates.copy().reset_index(drop=True)
    history = history_edges.sort_values("time").reset_index(drop=True)
    if len(candidates) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros(0, dtype=np.float32)

    order = np.argsort(candidates["time"].to_numpy())
    feats = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=np.float32)
    heur = np.zeros(len(candidates), dtype=np.float32)
    hidx = 0
    hsrc = history["src"].to_numpy(dtype=np.int64) if len(history) else np.asarray([], dtype=np.int64)
    hdst = history["dst"].to_numpy(dtype=np.int64) if len(history) else np.asarray([], dtype=np.int64)
    htime = history["time"].to_numpy(dtype=np.float64) if len(history) else np.asarray([], dtype=np.float64)
    csrc = candidates["src"].to_numpy(dtype=np.int64)
    cdst = candidates["dst"].to_numpy(dtype=np.int64)
    ctime = candidates["time"].to_numpy(dtype=np.float64)
    for ci in order:
        ci = int(ci)
        t = float(ctime[ci])
        while hidx < len(history) and float(htime[hidx]) < t:
            builder.add_edge(int(hsrc[hidx]), int(hdst[hidx]), float(htime[hidx]))
            hidx += 1
        feats[ci], heur[ci] = builder.score(int(csrc[ci]), int(cdst[ci]), t)
    return feats, heur


def build_feature_builder(history_edges: pd.DataFrame, window: float = 7.0) -> CausalFeatureBuilder:
    builder = CausalFeatureBuilder(window=window)
    history = history_edges.sort_values("time").reset_index(drop=True)
    src = history["src"].to_numpy(dtype=np.int64)
    dst = history["dst"].to_numpy(dtype=np.int64)
    time = history["time"].to_numpy(dtype=np.float64)
    for i in range(len(history)):
        builder.add_edge(int(src[i]), int(dst[i]), float(time[i]))
    return builder


def score_candidate_features_from_builder(candidates: pd.DataFrame, builder: CausalFeatureBuilder) -> Tuple[np.ndarray, np.ndarray]:
    candidates = candidates.copy().reset_index(drop=True)
    feats = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=np.float32)
    heur = np.zeros(len(candidates), dtype=np.float32)
    if len(candidates) == 0:
        return feats, heur
    order = np.argsort(candidates["time"].to_numpy(dtype=np.float64))
    src = candidates["src"].to_numpy(dtype=np.int64)
    dst = candidates["dst"].to_numpy(dtype=np.int64)
    time = candidates["time"].to_numpy(dtype=np.float64)
    for ci in order:
        ci = int(ci)
        feats[ci], heur[ci] = builder.score(int(src[ci]), int(dst[ci]), float(time[ci]))
    return feats, heur


def build_training_pairs(
    train_edges: pd.DataFrame,
    num_negatives: int = 5,
    seed: int = 42,
    window: float = 7.0,
    max_events: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    edges = train_edges.sort_values("time").reset_index(drop=True)
    if max_events is not None and max_events > 0:
        edges = edges.iloc[-max_events:].reset_index(drop=True)
    src_arr = edges["src"].to_numpy(dtype=np.int64)
    dst_arr = edges["dst"].to_numpy(dtype=np.int64)
    time_arr = edges["time"].to_numpy(dtype=np.float64)
    dst_values, dst_counts = np.unique(dst_arr, return_counts=True)
    prob = np.power(dst_counts.astype(np.float64), 0.75)
    prob = prob / prob.sum()
    builder = CausalFeatureBuilder(window=window)
    seen_by_src = defaultdict(set)
    all_positive_by_src = defaultdict(set)
    for src, dst in zip(src_arr, dst_arr):
        all_positive_by_src[int(src)].add(int(dst))

    srcs, pos_dsts, ts = [], [], []
    neg_dsts = []
    pos_feats, neg_feats = [], []
    pos_heur, neg_heur = [], []

    for i in range(len(edges)):
        src, dst, t = int(src_arr[i]), int(dst_arr[i]), float(time_arr[i])
        pf, ph = builder.score(src, dst, t)
        ndsts: List[int] = []
        nf_list: List[np.ndarray] = []
        nh_list: List[float] = []
        tries = 0
        while len(ndsts) < num_negatives and tries < num_negatives * 100 + 100:
            ndst = int(rng.choice(dst_values, p=prob))
            tries += 1
            if ndst == dst or ndst in seen_by_src[src] or ndst in all_positive_by_src[src]:
                continue
            nf, nh = builder.score(src, ndst, t)
            ndsts.append(ndst)
            nf_list.append(nf)
            nh_list.append(nh)
        if len(ndsts) == num_negatives:
            srcs.append(src)
            pos_dsts.append(dst)
            ts.append(t)
            neg_dsts.append(ndsts)
            pos_feats.append(pf)
            neg_feats.append(nf_list)
            pos_heur.append(ph)
            neg_heur.append(nh_list)
        builder.add_edge(src, dst, t)
        seen_by_src[src].add(dst)

    return {
        "src": np.asarray(srcs, dtype=np.int64),
        "pos_dst": np.asarray(pos_dsts, dtype=np.int64),
        "neg_dst": np.asarray(neg_dsts, dtype=np.int64),
        "time": np.asarray(ts, dtype=np.float32),
        "pos_feat": np.asarray(pos_feats, dtype=np.float32),
        "neg_feat": np.asarray(neg_feats, dtype=np.float32),
        "pos_heur": np.asarray(pos_heur, dtype=np.float32),
        "neg_heur": np.asarray(neg_heur, dtype=np.float32),
    }
