from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import jittor as jt

        jt.set_global_seed(seed)
    except Exception:
        pass


def batch_indices(n: int, batch_size: int, shuffle: bool = True, seed: int = 42) -> Iterator[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def save_json(path: str | os.PathLike, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
