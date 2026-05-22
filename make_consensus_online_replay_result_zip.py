from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import jittor as jt

from make_combo_jittor_result_zip import _candidate_pool, _prob
from make_online_combo_jittor_result_zip import _feature_matrix_for_row, _update_online_stats
from make_temporal_result_zip import _build_stats as build_temporal_stats
from make_assoc_result_zip import build_assoc_stats
from sequential_heuristic import build_sequential_stats
from simple_heuristic import history_only
from track1_dynamic_rec.data import wide_candidate_columns
from track1_dynamic_rec.train_utils import load_json


def _read_teacher_dataset2(path: Path) -> np.ndarray:
    if path.is_dir():
        csv_path = path / "dataset2.csv"
        return pd.read_csv(csv_path, header=None).to_numpy(np.float64)
    with zipfile.ZipFile(path) as zf:
        with zf.open("dataset2.csv") as f:
            return pd.read_csv(f, header=None).to_numpy(np.float64)


def _teacher_stats(teachers: list[np.ndarray]) -> dict[str, np.ndarray]:
    margins = []
    tops = []
    for mat in teachers:
        order = np.argsort(-mat, axis=1)
        top = order[:, 0].astype(np.int16)
        margin = (np.take_along_axis(mat, order[:, :1], axis=1)[:, 0] -
                  np.take_along_axis(mat, order[:, 1:2], axis=1)[:, 0])
        tops.append(top)
        margins.append(margin.astype(np.float64))
    return {"tops": np.stack(tops, axis=1), "margins": np.stack(margins, axis=1)}


def _select_consensus_updates(
    row_id: int,
    candidates: np.ndarray,
    teachers: list[np.ndarray],
    stats: dict[str, np.ndarray],
    margin_thresholds: np.ndarray,
    min_top1_votes: int,
    min_topk_votes: int,
    topk: int,
    require_strong_teachers: int,
    max_updates: int,
) -> list[tuple[int, float]]:
    row_scores = [teacher[row_id] for teacher in teachers]
    strong = stats["margins"][row_id] >= margin_thresholds
    if int(strong.sum()) < require_strong_teachers:
        return []

    top1_votes = Counter(int(x) for x in stats["tops"][row_id])
    chosen: list[tuple[int, float]] = []
    for idx, votes in top1_votes.most_common():
        if votes >= min_top1_votes:
            avg_score = float(np.mean([scores[idx] for scores in row_scores]))
            chosen.append((idx, avg_score + 0.01 * votes))

    if len(chosen) < max_updates:
        topk_votes: Counter[int] = Counter()
        topk_scores: dict[int, list[float]] = {}
        for scores in row_scores:
            for rank, idx_raw in enumerate(np.argsort(-scores)[:topk]):
                idx = int(idx_raw)
                topk_votes[idx] += 1
                topk_scores.setdefault(idx, []).append(float(scores[idx]) / (rank + 1.0))
        for idx, votes in topk_votes.most_common():
            if votes < min_topk_votes or any(idx == old_idx for old_idx, _ in chosen):
                continue
            chosen.append((idx, float(np.mean(topk_scores[idx])) + 0.005 * votes))
            if len(chosen) >= max_updates:
                break

    chosen.sort(key=lambda x: x[1], reverse=True)
    out = []
    for idx, conf in chosen[:max_updates]:
        if 0 <= idx < len(candidates):
            out.append((int(candidates[idx]), float(conf)))
    return out


def _write_dataset2(
    data_root: Path,
    model_dir: Path,
    teacher_paths: list[Path],
    output_csv: Path,
    include_valid_history: bool,
    mode: str,
    temperature: float,
    min_top1_votes: int,
    min_topk_votes: int,
    topk: int,
    require_strong_teachers: int,
    margin_quantile: float,
    max_updates: int,
) -> tuple[int, dict[str, float]]:
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
    teachers = [_read_teacher_dataset2(path) for path in teacher_paths]
    if not teachers:
        raise ValueError("at least one teacher is required")
    if any(mat.shape[1] != 100 for mat in teachers):
        raise ValueError([mat.shape for mat in teachers])
    if len({mat.shape[0] for mat in teachers}) != 1:
        raise ValueError([mat.shape for mat in teachers])
    teacher_stats = _teacher_stats(teachers)
    margin_thresholds = np.quantile(teacher_stats["margins"], margin_quantile, axis=0)

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

    test = pd.read_csv(test_path)
    if len(test) != teachers[0].shape[0]:
        raise ValueError(f"teacher rows {teachers[0].shape[0]} != test rows {len(test)}")
    test = test.assign(_row_id=test.index.to_numpy(np.int64)).sort_values(["time", "_row_id"]).reset_index(drop=True)
    out_rows: list[list[str] | None] = [None] * len(test)
    update_buffer: list[tuple[int, int, float, float]] = []
    rows = 0
    pseudo_edges = 0
    selected_rows = 0
    last_time = None
    w_jt = jt.array(weights.reshape((-1, 1)))
    b_jt = jt.array(np.asarray([bias], dtype=np.float32))

    print(
        {
            "dataset": dataset,
            "history_edges": len(train),
            "candidate_pool": len(candidate_pool),
            "teachers": [str(x) for x in teacher_paths],
            "margin_quantile": margin_quantile,
            "margin_thresholds": [float(x) for x in margin_thresholds],
            "min_top1_votes": min_top1_votes,
            "min_topk_votes": min_topk_votes,
            "topk": topk,
            "require_strong_teachers": require_strong_teachers,
            "max_updates": max_updates,
        },
        flush=True,
    )

    def flush_updates() -> None:
        nonlocal pseudo_edges
        if not update_buffer:
            return
        update_buffer.sort(key=lambda x: x[3], reverse=True)
        for src_u, dst_u, time_u, _ in update_buffer:
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

    for _, row_obj in test.iterrows():
        time_value = float(row_obj["time"])
        if last_time is None:
            last_time = time_value
        if time_value != last_time:
            flush_updates()
            last_time = time_value

        row_id = int(row_obj["_row_id"])
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
        out_rows[row_id] = [f"{float(x):.8f}" for x in prob]

        updates = _select_consensus_updates(
            row_id,
            candidates,
            teachers,
            teacher_stats,
            margin_thresholds,
            min_top1_votes,
            min_topk_votes,
            topk,
            require_strong_teachers,
            max_updates,
        )
        if updates:
            selected_rows += 1
        for dst, conf in updates:
            update_buffer.append((src, dst, time_value, conf))

        rows += 1
        if rows == 1 or rows % 20000 == 0:
            print({"dataset": dataset, "rows": rows, "selected_rows": selected_rows, "pseudo_edges": pseudo_edges}, flush=True)
    flush_updates()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in out_rows:
            if row is None:
                raise RuntimeError("missing output row")
            writer.writerow(row)
    stats = {"rows": float(rows), "selected_rows": float(selected_rows), "pseudo_edges": float(pseudo_edges)}
    print({"dataset": dataset, **stats}, flush=True)
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/official_raw")
    parser.add_argument("--model-dir", type=str, default="outputs/combo_jittor")
    parser.add_argument("--baseline-dir", type=str, default="outputs/website_submission_97_major_dual_mlp_strong_boost")
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "row_sigmoid", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--min-top1-votes", type=int, default=3)
    parser.add_argument("--min-topk-votes", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--require-strong-teachers", type=int, default=3)
    parser.add_argument("--margin-quantile", type=float, default=0.75)
    parser.add_argument("--max-updates", type=int, default=1)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.baseline_dir) / "dataset1.csv", output_dir / "dataset1.csv")
    rows2, stats = _write_dataset2(
        Path(args.data_root),
        Path(args.model_dir),
        [Path(x) for x in args.teacher],
        output_dir / "dataset2.csv",
        include_valid_history=args.include_valid_history,
        mode=args.mode,
        temperature=args.temperature,
        min_top1_votes=args.min_top1_votes,
        min_topk_votes=args.min_topk_votes,
        topk=args.topk,
        require_strong_teachers=args.require_strong_teachers,
        margin_quantile=args.margin_quantile,
        max_updates=args.max_updates,
    )
    rows = {
        "dataset1": sum(1 for _ in (output_dir / "dataset1.csv").open("r", encoding="utf-8")),
        "dataset2": rows2,
    }
    zip_path = output_dir / "result.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_dir / "dataset1.csv", arcname="dataset1.csv")
        zf.write(output_dir / "dataset2.csv", arcname="dataset2.csv")
    print({"zip": str(zip_path), "rows": rows, "stats": stats})


if __name__ == "__main__":
    main()
