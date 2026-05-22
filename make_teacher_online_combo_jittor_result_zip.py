from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
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


def _read_teacher_dataset2(teacher_zip: Path) -> np.ndarray:
    with zipfile.ZipFile(teacher_zip) as zf:
        with zf.open("dataset2.csv") as f:
            teacher = pd.read_csv(f, header=None).to_numpy(np.float64)
    if teacher.shape[1] != 100:
        raise ValueError(f"teacher dataset2 expected 100 columns, got {teacher.shape}")
    return teacher


def _write_dataset2(
    data_root: Path,
    model_dir: Path,
    teacher_zip: Path,
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
    teacher = _read_teacher_dataset2(teacher_zip)

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
    if len(test) != teacher.shape[0]:
        raise ValueError(f"teacher rows {teacher.shape[0]} != test rows {len(test)}")
    test = test.assign(_row_id=test.index.to_numpy(np.int64)).sort_values(["time", "_row_id"]).reset_index(drop=True)
    out_rows: list[list[str] | None] = [None] * len(test)

    update_buffer: list[tuple[int, int, float, float]] = []
    margin_values: list[float] = []
    last_time = None
    pseudo_edges = 0
    rows = 0
    w_jt = jt.array(weights.reshape((-1, 1)))
    b_jt = jt.array(np.asarray([bias], dtype=np.float32))

    print(
        {
            "dataset": dataset,
            "history_edges": len(train),
            "candidate_pool": len(candidate_pool),
            "teacher_zip": str(teacher_zip),
            "update_topk": update_topk,
            "teacher_margin_quantile": margin_quantile,
        },
        flush=True,
    )

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

        teacher_row = teacher[row_id]
        order = np.argsort(-teacher_row)
        margin = float(teacher_row[order[0]] - teacher_row[order[1]]) if len(order) > 1 else 0.0
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
    parser.add_argument("--baseline-dir", type=str, default="outputs/website_submission_80_major_top4q85_35")
    parser.add_argument("--teacher-zip", type=str, default="submissions/80_major_top4q85_35/result.zip")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rank", choices=["rank", "row_sigmoid", "minmax"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-valid-history", action="store_true")
    parser.add_argument("--update-topk", type=int, default=4)
    parser.add_argument("--margin-quantile", type=float, default=0.80)
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
            Path(args.teacher_zip),
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
