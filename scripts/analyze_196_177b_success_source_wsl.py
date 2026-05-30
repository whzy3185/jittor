import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from wsl_retrieval_refine_utils import build_retrieval_features, load_bases, repo_root, topk_indices, write_json


def bucket_stats(name, values, changed, shift):
    q = np.quantile(values[~np.isnan(values)], [0, 0.25, 0.5, 0.75, 1.0]) if np.any(~np.isnan(values)) else np.array([0, 1, 2, 3, 4])
    rows = []
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        mask = (values >= lo) & (values <= hi if i == 3 else values < hi)
        rows.append({"bucket": f"{name}_q{i+1}", "rows": int(mask.sum()), "changed_rows": int((changed & mask).sum()), "mean_shift": float(shift[mask].mean()) if mask.any() else 0.0})
    return rows


def main():
    root = repo_root()
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    base177, base143, _ = load_bases(root)
    raw168_path = root / "outputs" / "website_submission_168_real_retrieval_cooccurrence_expert" / "dataset2.csv"
    raw168 = pd.read_csv(raw168_path, header=None).to_numpy(np.float32) if raw168_path.exists() else None
    f = build_retrieval_features(root)
    delta = np.abs(base177 - base143)
    changed = np.any(delta > 5e-9, axis=1)
    shift = delta.mean(axis=1)
    top3_change = np.array([set(a) != set(b) for a, b in zip(topk_indices(base143, 3), topk_indices(base177, 3))])
    top5_change = np.array([set(a) != set(b) for a, b in zip(topk_indices(base143, 5), topk_indices(base177, 5))])
    avoided = None
    if raw168 is not None:
        avoided = (np.argmax(raw168, axis=1) != np.argmax(base143, axis=1)) & (np.argmax(base177, axis=1) == np.argmax(base143, axis=1))
    out = {
        "id": 196,
        "name": "177b_success_source_wsl",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "changed_rows_vs_143": int(changed.sum()),
        "dataset2_mad_vs_143": float(delta.mean()),
        "top1_change_vs_143": float(np.mean(np.argmax(base177, axis=1) != np.argmax(base143, axis=1))),
        "top3_change_rate_vs_143": float(top3_change.mean()),
        "top5_change_rate_vs_143": float(top5_change.mean()),
        "raw168_top1_change_avoided_rows": int(avoided.sum()) if avoided is not None else None,
        "by_src_history_count": bucket_stats("src_history", f["src_history_count"].astype(float), changed, shift),
        "by_time": bucket_stats("time", f["test_time"].astype(float), changed, shift),
        "by_retrieval_margin": bucket_stats("retrieval_margin", f["retrieval_margin"].astype(float), changed, shift),
        "gain_source": "top1_guarded real-data retrieval score redistribution; raw 168 top1 errors are avoided while non-top1 candidate ranks move broadly",
        "top1_guard_is_key": True,
        "can_amplify": "cautiously; only top1_guard or high-confidence/time/history masks should be explored",
        "recommended_refinement": "generate 197 around w=0.075..0.20 with top1 guard, confidence masks, time decay, source-history buckets, and candidate-mask variants",
    }
    write_json(analysis / "196_177b_success_source_wsl.json", out)
    lines = ["# 196_177b_success_source_wsl", "", f"- changed_rows_vs_143: {out['changed_rows_vs_143']}", f"- dataset2_mad_vs_143: {out['dataset2_mad_vs_143']}", f"- top1_change_vs_143: {out['top1_change_vs_143']}", f"- raw168_top1_change_avoided_rows: {out['raw168_top1_change_avoided_rows']}", f"- gain_source: {out['gain_source']}", f"- can_amplify: {out['can_amplify']}"]
    (analysis / "196_177b_success_source_wsl.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"changed_rows": out["changed_rows_vs_143"], "mad": out["dataset2_mad_vs_143"], "can_amplify": out["can_amplify"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
