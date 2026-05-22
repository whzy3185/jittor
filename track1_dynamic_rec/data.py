from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SRC_ALIASES = ("src", "source", "user", "user_id", "u", "from", "from_id")
DST_ALIASES = ("dst", "target", "item", "item_id", "v", "to", "to_id")
TIME_ALIASES = ("time", "timestamp", "ts", "t", "date")
LABEL_ALIASES = ("label", "y", "target_label", "is_positive", "clicked")
QUERY_ALIASES = ("query_id", "qid", "user_time_id", "event_id", "interaction_id", "src_time")
SPLIT_ALIASES = ("split", "phase", "set", "mask")


@dataclass
class Schema:
    src: str = "src"
    dst: str = "dst"
    time: Optional[str] = "time"
    label: Optional[str] = "label"
    query_id: Optional[str] = "query_id"
    split: Optional[str] = "split"
    candidate_id: Optional[str] = "candidate_id"


@dataclass
class DatasetBundle:
    train_edges: pd.DataFrame
    val_candidates: Optional[pd.DataFrame]
    test_candidates: Optional[pd.DataFrame]
    all_edges: pd.DataFrame
    schema: Schema
    num_nodes: int
    num_src_nodes: int
    num_dst_nodes: int
    data_dir: Path
    sample_submission: Optional[Path]
    notes: List[str]
    id_mapping: Dict[int, int]


def _find_col(columns: Iterable[str], aliases: Tuple[str, ...]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    for c in columns:
        lc = c.lower()
        if any(alias in lc for alias in aliases):
            return c
    return None


def _read_table(path: Path, nrows: int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows if nrows and nrows > 0 else None)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep=None, engine="python", nrows=nrows if nrows and nrows > 0 else None)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            if all(isinstance(v, dict) for v in obj.values()):
                return pd.DataFrame.from_dict(obj, orient="index").reset_index(names="candidate_id")
            return pd.DataFrame({"candidate_id": list(obj.keys()), "value": list(obj.values())})
    if suffix == ".pkl":
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            return pd.DataFrame(obj)
    raise ValueError(f"Unsupported table file: {path}")


def _normalize_split_value(x) -> str:
    if isinstance(x, str):
        lx = x.strip().lower()
        if lx in ("0", "train", "training"):
            return "train"
        if lx in ("1", "valid", "val", "validation"):
            return "valid"
        if lx in ("2", "test", "testing"):
            return "test"
        return lx
    if int(x) == 0:
        return "train"
    if int(x) == 1:
        return "valid"
    if int(x) == 2:
        return "test"
    return str(x)


def _standardize_frame(df: pd.DataFrame, source_name: str, require_label: bool = False) -> Tuple[pd.DataFrame, Schema]:
    df = df.copy()
    schema = Schema()
    candidate_cols = [c for c in df.columns if str(c).lower().startswith("c") and str(c)[1:].isdigit()]
    if "src" in {c.lower() for c in df.columns} and candidate_cols and _find_col(df.columns, DST_ALIASES) is None:
        src_col = _find_col(df.columns, SRC_ALIASES)
        time_col = _find_col(df.columns, TIME_ALIASES)
        if src_col is None:
            raise ValueError(f"{source_name} has candidate columns but no src column.")
        if time_col is None:
            df["time"] = np.arange(len(df), dtype=np.float64)
            time_col = "time"
        candidate_cols = sorted(candidate_cols, key=lambda x: int(str(x)[1:]))
        n = len(df)
        k = len(candidate_cols)
        src_values = pd.to_numeric(df[src_col], errors="raise").to_numpy(np.int64)
        time_values = pd.to_numeric(df[time_col], errors="coerce").fillna(0).to_numpy(np.float64)
        cand = df[candidate_cols].apply(pd.to_numeric, errors="raise").to_numpy(np.int64)
        query_idx = np.repeat(np.arange(n, dtype=np.int64), k)
        cand_rank = np.tile(np.arange(1, k + 1, dtype=np.int16), n)
        out = pd.DataFrame(
            {
                "src": np.repeat(src_values, k),
                "dst": cand.reshape(-1),
                "time": np.repeat(time_values, k),
                "query_id": query_idx.astype(str),
                "candidate_rank": cand_rank,
            }
        )
        out["candidate_id"] = out["query_id"] + "_" + out["candidate_rank"].astype(str)
        out["src"] = pd.to_numeric(out["src"], errors="raise").astype(np.int64)
        out["dst"] = pd.to_numeric(out["dst"], errors="raise").astype(np.int64)
        out["time"] = pd.to_numeric(out["time"], errors="coerce").fillna(0).astype(np.float64)
        schema.time = "time"
        schema.label = None
        schema.query_id = "query_id"
        schema.split = None
        schema.candidate_id = "candidate_id"
        return out, schema

    src = _find_col(df.columns, SRC_ALIASES)
    dst = _find_col(df.columns, DST_ALIASES)
    if src is None or dst is None:
        raise ValueError(f"{source_name} does not expose recognizable src/dst columns. Columns: {list(df.columns)}")
    time = _find_col(df.columns, TIME_ALIASES)
    label = _find_col(df.columns, LABEL_ALIASES)
    query_id = _find_col(df.columns, QUERY_ALIASES)
    split = _find_col(df.columns, SPLIT_ALIASES)

    rename = {src: "src", dst: "dst"}
    if time is not None:
        rename[time] = "time"
    if label is not None:
        rename[label] = "label"
    if query_id is not None:
        rename[query_id] = "query_id"
    if split is not None:
        rename[split] = "split"
    df = df.rename(columns=rename)

    if "time" not in df.columns:
        df["time"] = np.arange(len(df), dtype=np.float64)
    if "query_id" not in df.columns:
        df["query_id"] = df["src"].astype(str) + "_" + df["time"].astype(str)
    if "candidate_id" not in df.columns:
        df["candidate_id"] = np.arange(len(df), dtype=np.int64)
    if "label" not in df.columns and require_label:
        df["label"] = 1

    for col in ("src", "dst"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(np.int64)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(np.float64)
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.float32)
    if "split" in df.columns:
        df["split"] = df["split"].map(_normalize_split_value)

    schema.time = "time"
    schema.label = "label" if "label" in df.columns else None
    schema.query_id = "query_id"
    schema.split = "split" if "split" in df.columns else None
    schema.candidate_id = "candidate_id"
    return df, schema


def _candidate_files(data_dir: Path, kind: str) -> List[Path]:
    pats = []
    if kind == "valid":
        pats = ["*valid*cand*.csv", "*val*cand*.csv", "*valid*.csv", "*val*.csv", "*valid*.json", "*val*.json"]
    elif kind == "test":
        pats = ["*test*cand*.csv", "*test*.csv", "*test*.json"]
    files: List[Path] = []
    for pat in pats:
        files.extend(data_dir.rglob(pat))
    return sorted({p for p in files if "submission" not in p.name.lower() and p.is_file()})


def _main_edge_files(data_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for pat in ("*.csv", "*.tsv", "*.txt", "*.pkl"):
        candidates.extend(data_dir.rglob(pat))
    blocked = ("submission", "result", "pred", "candidate", "valid", "val", "test")
    out = [p for p in candidates if p.is_file() and not any(b in p.name.lower() for b in blocked)]
    return sorted(out, key=lambda p: (0 if "train" in p.name.lower() or "edge" in p.name.lower() else 1, len(str(p))))


def _sample_submission(data_dir: Path) -> Optional[Path]:
    for pat in ("*sample*submission*.csv", "*sample*submission*.json", "*submit*.csv", "*submission*.json"):
        files = [p for p in data_dir.rglob(pat) if p.is_file()]
        if files:
            return sorted(files, key=lambda p: len(str(p)))[0]
    return None


def wide_candidate_columns(columns: Iterable[str]) -> List[str]:
    return sorted(
        [c for c in columns if str(c).lower().startswith("c") and str(c)[1:].isdigit()],
        key=lambda x: int(str(x)[1:]),
    )


def find_test_candidate_file(data_dir: str | os.PathLike) -> Optional[Path]:
    files = _candidate_files(Path(data_dir), "test")
    return files[0] if files else None


def collect_wide_test_node_values(test_path: Path, max_queries: int = 0, chunksize: int = 4096) -> np.ndarray:
    header = pd.read_csv(test_path, nrows=0)
    candidate_cols = wide_candidate_columns(header.columns)
    src_col = _find_col(header.columns, SRC_ALIASES)
    if src_col is None or not candidate_cols:
        return np.asarray([], dtype=np.int64)
    usecols = [src_col] + candidate_cols
    values = set()
    read_kwargs = {
        "usecols": usecols,
        "chunksize": chunksize,
        "nrows": max_queries if max_queries and max_queries > 0 else None,
    }
    for chunk in pd.read_csv(test_path, **read_kwargs):
        arr = chunk.apply(pd.to_numeric, errors="raise").to_numpy(np.int64).reshape(-1)
        values.update(int(x) for x in arr)
    return np.asarray(sorted(values), dtype=np.int64)


def _remap_ids(
    train_edges: pd.DataFrame,
    frames: List[Optional[pd.DataFrame]],
    extra_values: Optional[List[np.ndarray]] = None,
) -> Tuple[pd.DataFrame, List[Optional[pd.DataFrame]], Dict[int, int]]:
    values = [train_edges["src"].to_numpy(), train_edges["dst"].to_numpy()]
    for frame in frames:
        if frame is not None and len(frame):
            values.extend([frame["src"].to_numpy(), frame["dst"].to_numpy()])
    if extra_values:
        values.extend([v for v in extra_values if v is not None and len(v)])
    unique = np.unique(np.concatenate(values).astype(np.int64))
    mapping = {int(v): int(i) + 1 for i, v in enumerate(unique)}

    def apply(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        df = df.copy()
        df["src_raw"] = df["src"]
        df["dst_raw"] = df["dst"]
        src = df["src"].map(mapping)
        dst = df["dst"].map(mapping)
        if src.isna().any() or dst.isna().any():
            raise ValueError("ID remapping missed at least one node. Check extra_values and candidate parsing.")
        df["src"] = src.astype(np.int64)
        df["dst"] = dst.astype(np.int64)
        return df

    return apply(train_edges), [apply(f) for f in frames], mapping


def _build_candidates_from_negatives(edges: pd.DataFrame, neg_path: Path, split_name: str) -> pd.DataFrame:
    neg = np.load(neg_path)
    if neg.ndim == 1:
        neg = neg.reshape(-1, 1)
    if len(edges) != neg.shape[0]:
        raise ValueError(f"{neg_path} rows {neg.shape[0]} do not match {split_name} edges {len(edges)}")
    rows = []
    for i, (_, edge) in enumerate(edges.reset_index(drop=True).iterrows()):
        qid = f"{split_name}_{i}"
        rows.append({"src": edge.src, "dst": edge.dst, "time": edge.time, "label": 1, "query_id": qid, "candidate_id": f"{qid}_pos"})
        for j, ndst in enumerate(neg[i]):
            rows.append({"src": edge.src, "dst": int(ndst), "time": edge.time, "label": 0, "query_id": qid, "candidate_id": f"{qid}_neg{j}"})
    return pd.DataFrame(rows)


def _make_validation_candidates(train_edges: pd.DataFrame, val_edges: pd.DataFrame, num_negatives: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dst_values = np.unique(train_edges["dst"].to_numpy())
    if len(dst_values) == 0:
        dst_values = np.unique(val_edges["dst"].to_numpy())
    seen_pairs = set(zip(train_edges["src"].astype(int), train_edges["dst"].astype(int)))
    rows = []
    for i, (_, edge) in enumerate(val_edges.reset_index(drop=True).iterrows()):
        qid = f"valid_{i}"
        rows.append({"src": int(edge.src), "dst": int(edge.dst), "time": float(edge.time), "label": 1, "query_id": qid, "candidate_id": f"{qid}_pos"})
        sampled = 0
        tries = 0
        while sampled < num_negatives and tries < num_negatives * 50 + 50:
            ndst = int(dst_values[rng.randint(0, len(dst_values))])
            tries += 1
            if ndst == int(edge.dst) or (int(edge.src), ndst) in seen_pairs:
                continue
            rows.append({"src": int(edge.src), "dst": ndst, "time": float(edge.time), "label": 0, "query_id": qid, "candidate_id": f"{qid}_neg{sampled}"})
            sampled += 1
    return pd.DataFrame(rows)


def load_competition_data(
    data_dir: str | os.PathLike,
    val_ratio: float = 0.1,
    num_val_negatives: int = 50,
    seed: int = 42,
    max_val_events: int = 0,
    max_test_queries: int = 0,
    load_test_candidates: bool = True,
    map_test_candidates: bool = True,
    test_map_chunksize: int = 4096,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    notes: List[str] = []

    main_files = _main_edge_files(data_dir)
    if not main_files:
        raise FileNotFoundError(f"No train/edge table found under {data_dir}. Put the official Track 1 data there.")
    main_df, schema = _standardize_frame(_read_table(main_files[0]), str(main_files[0]), require_label=True)
    notes.append(f"main_edges={main_files[0]}")

    if "split" in main_df.columns:
        train_edges = main_df[main_df["split"] == "train"].copy()
        val_edges = main_df[main_df["split"].isin(["valid", "val"])].copy()
        test_edges = main_df[main_df["split"] == "test"].copy()
    else:
        main_df = main_df.sort_values("time").reset_index(drop=True)
        cut = max(1, int(len(main_df) * (1.0 - val_ratio)))
        train_edges = main_df.iloc[:cut].copy()
        val_edges = main_df.iloc[cut:].copy()
        test_edges = pd.DataFrame(columns=main_df.columns)
        notes.append(f"no split column; temporal holdout val_ratio={val_ratio}")

    if max_val_events and max_val_events > 0 and len(val_edges) > max_val_events:
        val_edges = val_edges.sort_values("time").tail(max_val_events).copy()
        notes.append(f"limited validation edges={max_val_events}")

    val_candidates = None
    test_candidates = None
    test_node_values: List[np.ndarray] = []
    val_files = _candidate_files(data_dir, "valid")
    test_files = _candidate_files(data_dir, "test")
    if val_files:
        val_candidates, _ = _standardize_frame(_read_table(val_files[0]), str(val_files[0]), require_label=False)
        notes.append(f"valid_candidates={val_files[0]}")
    if test_files:
        if load_test_candidates:
            test_candidates, _ = _standardize_frame(_read_table(test_files[0], nrows=max_test_queries), str(test_files[0]), require_label=False)
            notes.append(f"test_candidates={test_files[0]}")
        elif map_test_candidates:
            test_node_values.append(collect_wide_test_node_values(test_files[0], max_queries=max_test_queries, chunksize=test_map_chunksize))
            notes.append(f"mapped_test_candidate_nodes={test_files[0]}")
        if max_test_queries and max_test_queries > 0:
            notes.append(f"limited test queries={max_test_queries}")

    for ns_name, split_edges, target in (("val_ns.npy", val_edges, "valid"), ("test_ns.npy", test_edges, "test")):
        paths = list(data_dir.rglob(f"*{ns_name}")) + list(data_dir.rglob(f"*_{target}_ns.npy"))
        if paths and len(split_edges):
            cand = _build_candidates_from_negatives(split_edges, paths[0], target)
            if target == "valid" and val_candidates is None:
                val_candidates = cand
            if target == "test" and test_candidates is None:
                test_candidates = cand
            notes.append(f"{target}_negative_samples={paths[0]}")

    if val_candidates is None and len(val_edges):
        val_candidates = _make_validation_candidates(train_edges, val_edges, num_val_negatives, seed)
        notes.append(f"generated validation negatives={num_val_negatives}")

    train_edges, remapped, id_mapping = _remap_ids(
        train_edges,
        [val_edges, test_edges, val_candidates, test_candidates, main_df],
        extra_values=test_node_values,
    )
    val_edges, test_edges, val_candidates, test_candidates, all_edges = remapped
    all_frames = [train_edges]
    if val_edges is not None and len(val_edges):
        all_frames.append(val_edges)
    all_edges = pd.concat(all_frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    num_nodes = len(id_mapping) + 1

    return DatasetBundle(
        train_edges=train_edges.sort_values("time").reset_index(drop=True),
        val_candidates=val_candidates,
        test_candidates=test_candidates,
        all_edges=all_edges,
        schema=schema,
        num_nodes=int(num_nodes),
        num_src_nodes=int(max(train_edges["src"].max(), 1)) + 1,
        num_dst_nodes=int(max(train_edges["dst"].max(), 1)) + 1,
        data_dir=data_dir,
        sample_submission=_sample_submission(data_dir),
        notes=notes,
        id_mapping=id_mapping,
    )


def describe_bundle(bundle: DatasetBundle) -> str:
    parts = [
        f"train_edges={len(bundle.train_edges)}",
        f"val_candidates={0 if bundle.val_candidates is None else len(bundle.val_candidates)}",
        f"test_candidates={0 if bundle.test_candidates is None else len(bundle.test_candidates)}",
        f"num_nodes={bundle.num_nodes}",
    ]
    if bundle.sample_submission:
        parts.append(f"sample_submission={bundle.sample_submission}")
    if bundle.notes:
        parts.append("notes=" + "; ".join(bundle.notes))
    return " | ".join(parts)
