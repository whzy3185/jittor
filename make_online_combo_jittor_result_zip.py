from __future__ import annotations

import argparse
import csv
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt

from enhanced_heuristic import _pair_key
from make_assoc_result_zip import _pack, assoc_component_matrix, build_assoc_stats
from make_combo_jittor_result_zip import _candidate_pool, _prob
from make_temporal_result_zip import _build_stats as build_temporal_stats
from make_temporal_result_zip import temporal_component_matrix
from sequential_heuristic import build_sequential_stats, sequential_component_matrix
from simple_heuristic import history_only
from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.train_utils import load_json


def _push_recent_array(recent: np.ndarray | None, dst: int, limit: int) -> np.ndarray:
    out = [int(dst)]
    if recent is not None:
        for value in recent:
            x = int(value)
            if x == int(dst):
                continue
            out.append(x)
            if len(out) >= limit:
                break
    return np.asarray(out[:limit], dtype=np.int64)


def _append_time(series: dict, key, value: float) -> None:
    old = series.get(key)
    if old is None or len(old) == 0:
        series[key] = np.asarray([float(value)], dtype=np.float64)
    else:
        series[key] = np.concatenate([old, np.asarray([float(value)], dtype=np.float64)])


def _update_online_stats(
    seq_stats,
    temporal_stats,
    assoc_stats,
    src: int,
    dst: int,
    time_value: float,
    *,
    max_history_per_src: int,
    max_pairs_per_src: int,
    assoc_context: int,
    assoc_history: int,
    candidate_pool: set[int],
) -> None:
    src = int(src)
    dst = int(dst)
    t = float(time_value)
    old_recent = seq_stats.enhanced.recent_by_src.get(src)

    if old_recent is not None and len(old_recent) > 0:
        prev = int(old_recent[0])
        if prev != dst:
            seq_stats.trans_count[(prev, dst)] = int(seq_stats.trans_count.get((prev, dst), 0)) + 1
            seq_stats.trans_out[prev] = int(seq_stats.trans_out.get(prev, 0)) + 1

    simple = seq_stats.enhanced.simple
    key = (src, dst)
    simple.dst_pop[dst] = int(simple.dst_pop.get(dst, 0)) + 1
    simple.src_degree[src] = int(simple.src_degree.get(src, 0)) + 1
    simple.pair_count[key] = int(simple.pair_count.get(key, 0)) + 1
    simple.last_pair[key] = t
    simple.last_dst[dst] = t
    simple.last_src[src] = t

    if old_recent is not None and len(old_recent) > 0:
        for prev_raw in old_recent[:max_pairs_per_src]:
            prev = int(prev_raw)
            if prev == dst:
                continue
            if dst in candidate_pool or prev in candidate_pool:
                k = _pair_key(prev, dst)
                seq_stats.enhanced.cooc[k] = int(seq_stats.enhanced.cooc.get(k, 0)) + 1
    seq_stats.enhanced.recent_by_src[src] = _push_recent_array(old_recent, dst, max_history_per_src)

    _append_time(temporal_stats.dst_times, dst, t)
    _append_time(temporal_stats.src_times, src, t)
    _append_time(temporal_stats.pair_times, key, t)
    temporal_stats.last_dst[dst] = t
    temporal_stats.last_src[src] = t
    temporal_stats.last_pair[key] = t
    temporal_stats.dst_count[dst] = int(temporal_stats.dst_count.get(dst, 0)) + 1
    temporal_stats.pair_count[key] = int(temporal_stats.pair_count.get(key, 0)) + 1

    assoc_recent = assoc_stats.recent_by_src.get(src)
    if assoc_recent is not None and len(assoc_recent) > 0 and dst in candidate_pool:
        for rank, prev_raw in enumerate(assoc_recent[:assoc_context]):
            prev = int(prev_raw)
            if prev == dst:
                continue
            weight = 1.0 / math.log2(rank + 2.0)
            assoc_key = _pack(prev, dst)
            assoc_stats.assoc[assoc_key] = float(assoc_stats.assoc.get(assoc_key, 0.0)) + weight
            assoc_stats.assoc_out[prev] = float(assoc_stats.assoc_out.get(prev, 0.0)) + weight
    assoc_stats.dst_pop[dst] = int(assoc_stats.dst_pop.get(dst, 0)) + 1
    assoc_stats.pair_count[key] = int(assoc_stats.pair_count.get(key, 0)) + 1
    assoc_stats.recent_by_src[src] = _push_recent_array(assoc_recent, dst, assoc_history)


def _feature_matrix_for_row(
    src: int,
    time_value: float,
    candidates: np.ndarray,
    seq_stats,
    temporal_stats,
    assoc_stats,
    windows: list[float],
    taus: list[float],
    decay_cap: int,
    assoc_score_history: int,
) -> np.ndarray:
    sfeat = sequential_component_matrix(src, time_value, candidates, seq_stats)
    tfeat = temporal_component_matrix(src, time_value, candidates, temporal_stats, windows, taus, decay_cap)
    afeat = assoc_component_matrix(src, candidates, assoc_stats, assoc_score_history)
    return np.concatenate([sfeat, tfeat, afeat], axis=1).astype(np.float32)


def _write_dataset2(
    data_root: Path,
    model_dir: Path,
    output_csv: Path,
    include_valid_history: bool,
    mode: str,
    temperature: float,
    update_topk: int,
    margin_quantile: float,
) -> int:
    dataset = "dataset2"
    meta = load_json(model_dir / "dataset2_combo_jittor.json")
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    weights = np.asarray(meta["weights"], dtype=np.float32)
    bias = float(meta.get("bias", 0.0))
    args = meta["args"]
    windows = [float(x) for x in meta["windows"]]
    taus = [float(x) for x in meta["taus"]]

    data_dir = data_root / dataset
    train = history_only(pd.read_csv(data_dir / "train.csv"), include_valid_history=include_valid_history)
    test_path = data_dir / "test.csv"
    cand_cols = wide_candidate_columns(pd.read_csv(test_path, nrows=0).columns)
    candidate_pool = _candidate_pool(test_path, cand_cols)
    seq_stats = build_sequential_stats(
        train,
        candidate_pool=candidate_pool,
        max_history_per_src=int(args["seq_history"]),
        max_pairs_per_src=int(args["seq_pairs"]),
    )
    temporal_stats = build_temporal_stats(train, candidate_pool=candidate_pool, keep_pair_times=True)
    assoc_stats = build_assoc_stats(
        train,
        candidate_pool=candidate_pool,
        max_context=int(args["assoc_context"]),
        max_history_per_src=int(args["assoc_history"]),
        min_count=float(args["assoc_min_count"]),
    )
    print(
        {
            "dataset": dataset,
            "history_edges": len(train),
            "candidate_pool": len(candidate_pool),
            "mode": mode,
            "update_topk": update_topk,
            "margin_quantile": margin_quantile,
        },
        flush=True,
    )

    test = pd.read_csv(test_path)
    original_index = test.index.to_numpy(np.int64)
    test = test.assign(_row_id=original_index).sort_values(["time", "_row_id"]).reset_index(drop=True)
    out_rows: list[list[str] | None] = [None] * len(test)
    update_buffer: list[tuple[int, int, float, float]] = []
    margin_values: list[float] = []
    last_time = None
    pseudo_edges = 0
    rows = 0
    w_jt = jt.array(weights.reshape((-1, 1)))
    b_jt = jt.array(np.asarray([bias], dtype=np.float32))

    def flush_updates() -> None:
        nonlocal pseudo_edges
        if not update_buffer:
            return
        if margin_values and 0.0 < margin_quantile < 1.0:
            threshold = float(np.quantile(np.asarray(margin_values, dtype=np.float64), margin_quantile))
        else:
            threshold = -float("inf")
        for src_u, dst_u, time_u, margin_u in update_buffer:
            if margin_u >= threshold:
                _update_online_stats(
                    seq_stats,
                    temporal_stats,
                    assoc_stats,
                    src_u,
                    dst_u,
                    time_u,
                    max_history_per_src=int(args["seq_history"]),
                    max_pairs_per_src=int(args["seq_pairs"]),
                    assoc_context=int(args["assoc_context"]),
                    assoc_history=int(args["assoc_history"]),
                    candidate_pool=candidate_pool,
                )
                pseudo_edges += 1
        update_buffer.clear()
        margin_values.clear()

    for _, row_obj in test.iterrows():
        time_value = float(row_obj["time"])
        if last_time is None:
            last_time = time_value
        if time_value != last_time:
            flush_updates()
            last_time = time_value
        src = int(row_obj["src"])
        candidates = row_obj[cand_cols].to_numpy(np.int64)
        feat = _feature_matrix_for_row(
            src,
            time_value,
            candidates,
            seq_stats,
            temporal_stats,
            assoc_stats,
            windows,
            taus,
            int(args["decay_cap"]),
            int(args["assoc_score_history"]),
        )
        feat = ((feat - mean) / std).astype(np.float32)
        raw = (jt.array(feat).matmul(w_jt) + b_jt).numpy().reshape(-1)
        prob = np.clip(_prob(raw, mode, temperature), 0.0, 1.0)
        out_rows[int(row_obj["_row_id"])] = [f"{float(x):.8f}" for x in prob]
        order = np.argsort(-raw)
        margin = float(raw[order[0]] - raw[order[1]]) if len(order) > 1 else 0.0
        margin_values.append(margin)
        for idx in order[: max(1, int(update_topk))]:
            update_buffer.append((src, int(candidates[int(idx)]), time_value, margin))
        rows += 1
        if rows == 1 or rows % 20000 == 0:
            print({"dataset": dataset, "rows": rows, "pseudo_edges": pseudo_edges}, flush=True)
    flush_updates()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in out_rows:
            if row is None:
                raise RuntimeError("missing output row")
            writer.writerow(row)
    print({"dataset": dataset, "rows": rows, "pseudo_edges": pseudo_edges}, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/combo_jittor")
    parser.add_argument("--baseline-dir", type=str, default="outputs/website_submission_69_d1_62_d2_online07_combo36_temp06")
    parser.add_argument("--output-dir", type=str, default="outputs/website_submission_online_combo_jittor")
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "row_sigmoid", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--update-topk", type=int, default=1)
    parser.add_argument("--margin-quantile", type=float, default=0.5)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.baseline_dir) / "dataset1.csv", output_dir / "dataset1.csv")
    rows = {
        "dataset1": sum(1 for _ in (output_dir / "dataset1.csv").open("r", encoding="utf-8")),
        "dataset2": _write_dataset2(
            Path(args.data_root),
            Path(args.model_dir),
            output_dir / "dataset2.csv",
            include_valid_history=args.include_valid_history,
            mode=args.mode,
            temperature=args.temperature,
            update_topk=args.update_topk,
            margin_quantile=args.margin_quantile,
        ),
    }
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows})


if __name__ == "__main__":
    main()
