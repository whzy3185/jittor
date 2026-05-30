import json
from pathlib import Path

import numpy as np

from wsl_retrieval_refine_utils import build_retrieval_features, ensure_dir, load_bases, repo_root, topk_indices, write_candidate, write_json


def blend(base, signal, weight):
    return np.clip((1.0 - weight) * base + weight * signal, 0.0, 1.0)


def apply(base, signal, mask, weight, guard):
    prop = blend(base, signal, weight)
    final_mask = mask.copy()
    base_top1 = np.argmax(base, axis=1)
    if guard == "top1":
        final_mask &= np.argmax(prop, axis=1) == base_top1
    elif guard == "top5":
        new_top1 = np.argmax(prop, axis=1)
        final_mask &= np.any(topk_indices(base, 5) == new_top1[:, None], axis=1)
    out = base.copy()
    out[final_mask] = prop[final_mask]
    return out, int(final_mask.sum())


def main():
    root = repo_root()
    analysis = root / "analysis"
    ensure_dir(analysis)
    base177, base143, _ = load_bases(root)
    f = build_retrieval_features(root)
    rrank = f["retrieval_rank"]
    decay_rank = f["retrieval_decay_rank"]
    margin = f["retrieval_margin"]
    support = f["hist_support_count"]
    hist = f["src_history_count"]
    tm = f["test_time"]
    entropy = f["retrieval_entropy"]
    q90, q95, q975, q99 = [float(np.quantile(margin, q)) for q in [0.90, 0.95, 0.975, 0.99]]
    t50, t75 = [float(np.quantile(tm, q)) for q in [0.50, 0.75]]
    h2, h3, h5 = hist >= 2, hist >= 3, hist >= 5
    pop = f["dst_count_raw"]
    cand = f["candidates"]
    rtop = f["retrieval_top5"][:, 0]
    rclass = cand[np.arange(len(cand)), rtop]
    pop_score = pop[np.minimum(rclass, len(pop) - 1)]
    pop_dominated = pop_score >= np.quantile(pop_score, 0.95)

    specs = [
        ("197a", "w0075_top1_guard", 0.075, np.ones(len(base177), bool), rrank, "top1"),
        ("197b", "w0125_top1_guard", 0.125, np.ones(len(base177), bool), rrank, "top1"),
        ("197c", "w0150_top1_guard", 0.150, np.ones(len(base177), bool), rrank, "top1"),
        ("197d", "w0200_top1_guard", 0.200, np.ones(len(base177), bool), rrank, "top1"),
        ("197e", "w010_conf_q90_top1_guard", 0.100, margin >= q90, rrank, "top1"),
        ("197f", "w010_conf_q95_top1_guard", 0.100, margin >= q95, rrank, "top1"),
        ("197g", "w010_conf_q975_top1_guard", 0.100, margin >= q975, rrank, "top1"),
        ("197h", "w010_conf_q99_top1_guard", 0.100, margin >= q99, rrank, "top1"),
        ("197i", "w0125_conf_q95_top1_guard", 0.125, margin >= q95, rrank, "top1"),
        ("197j", "w0150_conf_q95_top1_guard", 0.150, margin >= q95, rrank, "top1"),
        ("197k", "w0200_conf_q95_top1_guard", 0.200, margin >= q95, rrank, "top1"),
        ("197l", "time_decay_0p90", 0.090, support > 0, decay_rank, "top1"),
        ("197m", "time_decay_0p95", 0.095, support > 0, decay_rank, "top1"),
        ("197n", "time_decay_0p98", 0.098, support > 0, decay_rank, "top1"),
        ("197o", "src_history_ge2", 0.125, h2, rrank, "top1"),
        ("197p", "src_history_ge3", 0.150, h3, rrank, "top1"),
        ("197q", "src_history_ge5", 0.200, h5, rrank, "top1"),
        ("197r", "exclude_popularity_dominated", 0.150, ~pop_dominated, rrank, "top1"),
        ("197s", "candidate_only_retrieval", 0.125, support > 0, rrank, "top1"),
        ("197t", "no_global_popularity", 0.120, support > 0, decay_rank, "top1"),
        ("197u", "baseline_logic_time_decay", 0.140, (support > 0) & (tm >= t50), decay_rank, "top1"),
        ("197v", "baseline_logic_candidate_mask", 0.140, (support > 0) & (margin >= q90), rrank, "top1"),
        ("197w", "baseline_logic_graph_neighbor_if_available", 0.160, (support >= 2) | (hist >= 3), decay_rank, "top1"),
        ("197x", "final_balanced_wsl_retrieval", 0.125, ((margin >= q90) | (support > 0)) & (entropy <= np.quantile(entropy, 0.75)) & (tm >= t50), 0.5 * rrank + 0.5 * decay_rank, "top1"),
    ]
    summaries = []
    for cid, name, weight, mask, signal, guard in specs:
        mat, applied = apply(base177, signal.astype(np.float32), mask.astype(bool), weight, guard)
        summary = write_candidate(root, cid, name, mat, {
            "id": 197,
            "candidate_id": cid,
            "candidate_name": name,
            "base": "177b_w010_top1_guard",
            "weight": weight,
            "guard": guard,
            "mask_rows": int(mask.sum()),
            "applied_rows": applied,
            "risk_level": "low" if guard == "top1" else "medium",
        })
        summaries.append(summary)
    valid = [s for s in summaries if s["validate_pass"] and s["zip_root_pass"]]
    report = {"id": 197, "name": "wsl_refined_retrieval_grid", "generated_candidates": len(summaries), "valid_candidates": len(valid), "candidates": summaries}
    write_json(analysis / "197_wsl_refined_retrieval_grid.json", report)
    lines = ["# 197_wsl_refined_retrieval_grid", "", f"- generated_candidates: {len(summaries)}", f"- valid_candidates: {len(valid)}", "", "| candidate | mad_vs_177b | changed_vs_177b | top1_vs_177b | mad_vs_143 | top1_vs_143 | valid |", "|---|---:|---:|---:|---:|---:|---|"]
    for s in summaries:
        lines.append(f"| {s['candidate_id']}_{s['candidate_name']} | {s['dataset2_mad_vs_177b']:.8f} | {s['changed_rows_vs_177b']} | {s['top1_change_vs_177b']:.6f} | {s['dataset2_mad_vs_143']:.8f} | {s['top1_change_vs_143']:.6f} | {s['validate_pass'] and s['zip_root_pass']} |")
    (analysis / "197_wsl_refined_retrieval_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"generated": len(summaries), "valid": len(valid)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
