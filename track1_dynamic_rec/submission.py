from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


def write_predictions_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("candidate_id", "query_id", "candidate_rank", "src_raw", "dst_raw", "src", "dst", "time", "score") if c in df.columns]
    df[keep].to_csv(output_path, index=False)


def _write_wide_rank_files(df: pd.DataFrame, output_dir: Path) -> None:
    if not {"query_id", "dst_raw", "score"}.issubset(df.columns):
        return
    ranked_rows = []
    score_rows = []
    for qid, group in df.groupby("query_id", sort=False):
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        ranked_rows.append([qid] + [int(x) for x in ranked["dst_raw"].tolist()])
        score_order = group.sort_values("candidate_rank" if "candidate_rank" in group.columns else "candidate_id").reset_index(drop=True)
        score_rows.append([qid] + [float(x) for x in score_order["score"].tolist()])
    max_len = max((len(r) - 1 for r in ranked_rows), default=0)
    cols = ["query_id"] + [f"rank{i}" for i in range(1, max_len + 1)]
    pd.DataFrame(ranked_rows, columns=cols).to_csv(output_dir / "result_ranked.csv", index=False)
    score_cols = ["query_id"] + [f"c{i}" for i in range(1, max_len + 1)]
    pd.DataFrame(score_rows, columns=score_cols).to_csv(output_dir / "result_scores.csv", index=False)


def write_submission(df: pd.DataFrame, output_dir: str | Path, sample_submission: Optional[Path] = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / "result.json"
    predictions_csv = output_dir / "predictions.csv"
    write_predictions_csv(df, predictions_csv)
    _write_wide_rank_files(df, output_dir)

    if sample_submission is not None and sample_submission.exists() and sample_submission.suffix.lower() == ".csv":
        sample = pd.read_csv(sample_submission)
        score_col = "score" if "score" in sample.columns else ("prediction" if "prediction" in sample.columns else sample.columns[-1])
        out = sample.copy()
        n = min(len(out), len(df))
        out.loc[: n - 1, score_col] = df["score"].to_numpy()[:n]
        out.to_csv(output_dir / sample_submission.name, index=False)

    if {"query_id", "dst_raw", "score"}.issubset(df.columns):
        result = {}
        for qid, group in df.groupby("query_id", sort=False):
            ranked = group.sort_values("score", ascending=False)
            result[str(qid)] = [int(x) for x in ranked["dst_raw"].tolist()]
    else:
        key_col = "candidate_id" if "candidate_id" in df.columns else None
        result = {}
        for i, row in df.reset_index(drop=True).iterrows():
            key = str(row[key_col]) if key_col else str(i)
            result[key] = float(row.score)
    with result_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result_json


def package_submission(output_dir: str | Path, zip_path: str | Path) -> Path:
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("result.json", "predictions.csv", "result_ranked.csv", "result_scores.csv"):
            path = output_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    return zip_path
