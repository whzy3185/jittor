from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import jittor as jt

from track1_dynamic_rec.data import describe_bundle, find_test_candidate_file, load_competition_data, wide_candidate_columns
from track1_dynamic_rec.features import FeatureStats, build_feature_builder, resolve_time_window, score_candidate_features_from_builder
from track1_dynamic_rec.model import HybridTemporalScorer
from track1_dynamic_rec.train_utils import batch_indices, load_json


def _find_col(columns, aliases):
    lower = {str(c).lower(): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def _load_id_mapping(meta: dict, meta_path: Path) -> Optional[Dict[int, int]]:
    path = meta.get("id_mapping_path")
    if not path:
        return None
    mapping_path = Path(path)
    if not mapping_path.is_absolute():
        cwd_candidate = mapping_path.resolve()
        mapping_path = cwd_candidate if cwd_candidate.exists() else (meta_path.parent / mapping_path).resolve()
    if not mapping_path.exists():
        return None
    raw = load_json(mapping_path)
    return {int(k): int(v) for k, v in raw.items()}


def _rank01(values: np.ndarray) -> np.ndarray:
    if values.shape[1] <= 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(np.argsort(values, axis=1), axis=1)
    return order.astype(np.float32) / float(values.shape[1] - 1)


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "result_ranked.csv", "result_scores.csv", "predictions.csv"):
        target = path / name
        if target.exists():
            target.unlink()


def _write_zip(output_dir: Path) -> Path:
    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("result.json", "result_ranked.csv", "result_scores.csv", "predictions.csv"):
            path = output_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    return zip_path


def _score_flat(model, candidates: pd.DataFrame, stats: FeatureStats, builder, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    feat, heur = score_candidate_features_from_builder(candidates, builder)
    feat = stats.transform(feat).astype(np.float32)
    scores = np.zeros(len(candidates), dtype=np.float32)
    model.eval()
    for idx in batch_indices(len(candidates), batch_size, shuffle=False):
        src = jt.array(candidates.iloc[idx]["src"].to_numpy(np.int64))
        dst = jt.array(candidates.iloc[idx]["dst"].to_numpy(np.int64))
        t = jt.array(candidates.iloc[idx]["time"].to_numpy(np.float32))
        f = jt.array(feat[idx])
        scores[idx] = model.score(src, dst, t, f).detach().numpy()
    return scores, heur.astype(np.float32)


def stream_score_test(
    model,
    test_path: Path,
    output_dir: Path,
    id_mapping: Dict[int, int],
    history: pd.DataFrame,
    stats: FeatureStats,
    window: float,
    batch_size: int,
    query_chunk_size: int,
    model_weight: float,
    heuristic_weight: float,
    max_test_queries: int,
    write_debug_csv: bool,
) -> int:
    header = pd.read_csv(test_path, nrows=0)
    candidate_cols = wide_candidate_columns(header.columns)
    src_col = _find_col(header.columns, ("src", "source", "user", "user_id", "u"))
    time_col = _find_col(header.columns, ("time", "timestamp", "ts", "t"))
    if src_col is None or time_col is None or not candidate_cols:
        raise ValueError(f"{test_path} is not the official wide test format. Columns={list(header.columns)}")

    _prepare_output_dir(output_dir)
    builder = build_feature_builder(history, window=window)
    usecols = [src_col, time_col] + candidate_cols
    read_kwargs = {
        "usecols": usecols,
        "chunksize": query_chunk_size,
        "nrows": max_test_queries if max_test_queries and max_test_queries > 0 else None,
    }
    result_json = output_dir / "result.json"
    ranked_csv = output_dir / "result_ranked.csv"
    scores_csv = output_dir / "result_scores.csv"
    pred_csv = output_dir / "predictions.csv"
    total_queries = 0
    first_json = True

    with result_json.open("w", encoding="utf-8") as jf, ranked_csv.open("w", newline="", encoding="utf-8") as rf, scores_csv.open("w", newline="", encoding="utf-8") as sf:
        jf.write("{\n")
        ranked_writer = csv.writer(rf)
        scores_writer = csv.writer(sf)
        ranked_writer.writerow(["query_id"] + [f"rank{i}" for i in range(1, len(candidate_cols) + 1)])
        scores_writer.writerow(["query_id"] + [f"c{i}" for i in range(1, len(candidate_cols) + 1)])
        pred_handle = pred_csv.open("w", newline="", encoding="utf-8") if write_debug_csv else None
        pred_writer = csv.writer(pred_handle) if pred_handle else None
        if pred_writer:
            pred_writer.writerow(["query_id", "candidate_rank", "src_raw", "dst_raw", "time", "score", "model_score", "heuristic_score"])
        try:
            for chunk in pd.read_csv(test_path, **read_kwargs):
                n = len(chunk)
                k = len(candidate_cols)
                src_raw = pd.to_numeric(chunk[src_col], errors="raise").to_numpy(np.int64)
                time_values = pd.to_numeric(chunk[time_col], errors="coerce").fillna(0).to_numpy(np.float32)
                cand_raw = chunk[candidate_cols].apply(pd.to_numeric, errors="raise").to_numpy(np.int64)
                qids = np.arange(total_queries, total_queries + n, dtype=np.int64)
                src_ids = np.asarray([id_mapping.get(int(x), 0) for x in src_raw], dtype=np.int64)
                dst_ids = np.asarray([id_mapping.get(int(x), 0) for x in cand_raw.reshape(-1)], dtype=np.int64)
                candidates = pd.DataFrame(
                    {
                        "src": np.repeat(src_ids, k),
                        "dst": dst_ids,
                        "time": np.repeat(time_values, k),
                    }
                )
                model_score, heuristic_score = _score_flat(model, candidates, stats, builder, batch_size)
                model_mat = model_score.reshape(n, k)
                heur_mat = heuristic_score.reshape(n, k)
                score_mat = model_weight * _rank01(model_mat) + heuristic_weight * _rank01(heur_mat)
                ranked_idx = np.argsort(-score_mat, axis=1)
                for i in range(n):
                    qid = str(int(qids[i]))
                    ranked_items = [int(cand_raw[i, j]) for j in ranked_idx[i]]
                    if not first_json:
                        jf.write(",\n")
                    jf.write(f"  {json.dumps(qid)}: {json.dumps(ranked_items, ensure_ascii=False)}")
                    first_json = False
                    ranked_writer.writerow([qid] + ranked_items)
                    scores_writer.writerow([qid] + [f"{float(x):.8f}" for x in score_mat[i]])
                    if pred_writer:
                        for j in range(k):
                            pred_writer.writerow(
                                [
                                    qid,
                                    j + 1,
                                    int(src_raw[i]),
                                    int(cand_raw[i, j]),
                                    float(time_values[i]),
                                    f"{float(score_mat[i, j]):.8f}",
                                    f"{float(model_mat[i, j]):.8f}",
                                    f"{float(heur_mat[i, j]):.8f}",
                                ]
                            )
                total_queries += n
                print(json.dumps({"scored_queries": int(total_queries), "chunk_queries": int(n)}, ensure_ascii=False), flush=True)
        finally:
            if pred_handle:
                pred_handle.close()
        jf.write("\n}\n")
    return total_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--dataset", type=str, default="dataset1", choices=["dataset1", "dataset2"])
    parser.add_argument("--data-dir", type=str, default=None, help="Overrides --data-root/--dataset when set.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--meta", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/track1/submission_stream")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--max-val-events", type=int, default=0)
    parser.add_argument("--max-test-queries", type=int, default=0, help="Smoke-test query limit. Use 0 for full official test.")
    parser.add_argument("--window", type=float, default=None)
    parser.add_argument("--model-weight", type=float, default=None, help="Override checkpoint blend_model_weight.")
    parser.add_argument("--heuristic-weight", type=float, default=None, help="Override checkpoint blend_heuristic_weight.")
    parser.add_argument("--include-valid-in-test-history", action="store_true")
    parser.add_argument("--write-debug-csv", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    default_run_dir = Path("outputs/track1") / args.dataset
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_run_dir / "checkpoints" / "best.pkl"
    meta_path = Path(args.meta) if args.meta else default_run_dir / "checkpoints" / "best_meta.json"
    meta = load_json(meta_path)
    train_args = meta["args"]
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / args.dataset
    max_test_queries = args.max_test_queries or int(meta.get("max_test_queries", 0))
    bundle = load_competition_data(
        data_dir,
        num_val_negatives=int(train_args.get("val_negatives", 50)),
        seed=int(train_args.get("seed", 42)),
        max_val_events=args.max_val_events,
        max_test_queries=max_test_queries,
        load_test_candidates=False,
        map_test_candidates=True,
    )
    print(describe_bundle(bundle))
    id_mapping = _load_id_mapping(meta, meta_path) or bundle.id_mapping
    if int(meta["num_nodes"]) < (max(id_mapping.values()) if id_mapping else 0):
        raise ValueError("Checkpoint num_nodes is smaller than the loaded ID mapping. Use the mapping saved by the matching training run.")

    window = args.window if args.window is not None else float(meta.get("window", resolve_time_window(bundle.train_edges, 0.0)))
    time_shift = float(meta.get("time_shift", bundle.train_edges["time"].min() if len(bundle.train_edges) else 0.0))
    time_scale = float(meta.get("time_scale", max(1.0, window)))
    model_weight = float(args.model_weight) if args.model_weight is not None else float(meta.get("blend_model_weight", 0.82))
    heuristic_weight = float(args.heuristic_weight) if args.heuristic_weight is not None else float(meta.get("blend_heuristic_weight", 1.0 - model_weight))
    stats = FeatureStats.from_dict(meta["feature_stats"])
    model = HybridTemporalScorer(
        num_nodes=int(meta["num_nodes"]),
        num_features=int(meta["num_features"]),
        emb_dim=int(train_args.get("emb_dim", 128)),
        time_dim=int(train_args.get("time_dim", 32)),
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        num_layers=int(train_args.get("layers", 3)),
        dropout=float(train_args.get("dropout", 0.15)),
        time_shift=time_shift,
        time_scale=time_scale,
    )
    model.load_state_dict(jt.load(str(checkpoint)))
    test_path = find_test_candidate_file(data_dir)
    if test_path is None:
        raise FileNotFoundError(f"No official test candidate file found under {data_dir}")
    history = bundle.all_edges if args.include_valid_in_test_history else bundle.train_edges
    output_dir = Path(args.output_dir) / args.dataset if Path(args.output_dir).name in {"submission", "submission_stream"} else Path(args.output_dir)
    total = stream_score_test(
        model=model,
        test_path=test_path,
        output_dir=output_dir,
        id_mapping=id_mapping,
        history=history,
        stats=stats,
        window=window,
        batch_size=args.batch_size,
        query_chunk_size=args.query_chunk_size,
        model_weight=model_weight,
        heuristic_weight=heuristic_weight,
        max_test_queries=max_test_queries,
        write_debug_csv=args.write_debug_csv,
    )
    zip_path = _write_zip(output_dir)
    print(json.dumps({"wrote_queries": total, "output_dir": str(output_dir), "zip": str(zip_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
