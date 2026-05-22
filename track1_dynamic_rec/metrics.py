from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true > 0.5]
    neg = y_score[y_true <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    cmp = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(cmp + 0.5 * ties)


def ranking_metrics(df: pd.DataFrame, score_col: str = "score", label_col: str = "label", group_col: str = "query_id") -> Dict[str, float]:
    if label_col not in df.columns:
        return {}
    out = {"auc": _auc(df[label_col].to_numpy(), df[score_col].to_numpy())}
    mrrs, hits1, hits3, hits10, ndcg10 = [], [], [], [], []
    for _, group in df.groupby(group_col, sort=False):
        g = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        labels = g[label_col].to_numpy(dtype=np.float32)
        pos = np.where(labels > 0.5)[0]
        if len(pos) == 0:
            continue
        rank = int(pos[0]) + 1
        mrrs.append(1.0 / rank)
        hits1.append(float(rank <= 1))
        hits3.append(float(rank <= 3))
        hits10.append(float(rank <= 10))
        gains = labels[:10] / np.log2(np.arange(2, min(10, len(labels)) + 2))
        ideal = np.sort(labels)[::-1][:10] / np.log2(np.arange(2, min(10, len(labels)) + 2))
        denom = ideal.sum()
        ndcg10.append(float(gains.sum() / denom) if denom > 0 else 0.0)
    if mrrs:
        out.update(
            {
                "mrr": float(np.mean(mrrs)),
                "hits@1": float(np.mean(hits1)),
                "hits@3": float(np.mean(hits3)),
                "hits@10": float(np.mean(hits10)),
                "ndcg@10": float(np.mean(ndcg10)),
            }
        )
    return out


def rank_normalize(df: pd.DataFrame, score_col: str, group_col: str = "query_id") -> np.ndarray:
    norm = np.zeros(len(df), dtype=np.float32)
    for _, idx in df.groupby(group_col, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        values = df.iloc[idx][score_col].to_numpy()
        order = np.argsort(np.argsort(values))
        denom = max(1, len(order) - 1)
        norm[idx] = order.astype(np.float32) / denom
    return norm
