import json
from pathlib import Path

from wsl_retrieval_refine_utils import repo_root, write_json


def strict_ok(s):
    return (
        s.get("dataset1_mad_vs_177b") == 0
        and 0.001 <= s.get("dataset2_mad_vs_177b", 0) <= 0.025
        and s.get("changed_rows_vs_177b", 0) >= 10000
        and s.get("top1_change_vs_177b", 1) <= 0.01
        and s.get("validate_pass")
        and s.get("zip_root_pass")
        and s.get("risk_level") != "high"
        and s.get("dataset1_mad_vs_143") == 0
        and 0.01 <= s.get("dataset2_mad_vs_143", 0) <= 0.08
        and s.get("top1_change_vs_143", 1) <= 0.02
    )


def exploratory_ok(s):
    return (
        s.get("dataset1_mad_vs_177b") == 0
        and 0.001 <= s.get("dataset2_mad_vs_177b", 0) <= 0.04
        and s.get("changed_rows_vs_177b", 0) >= 10000
        and s.get("top1_change_vs_177b", 1) <= 0.02
        and s.get("validate_pass")
        and s.get("zip_root_pass")
        and s.get("real_data_derived_signal")
        and s.get("not_raw_168")
    )


def score(s):
    # Prefer modest nonzero movement from 177b with top1 unchanged and enough broad coverage.
    mad = s.get("dataset2_mad_vs_177b", 0)
    changed = s.get("changed_rows_vs_177b", 0)
    top1 = s.get("top1_change_vs_177b", 0)
    target = 0.006
    return (1.0 - min(abs(mad - target) / target, 1.0)) * 2.0 + min(changed / 80000, 1.0) - top1 * 20


def main():
    root = repo_root()
    analysis = root / "analysis"
    r197 = json.loads((analysis / "197_wsl_refined_retrieval_grid.json").read_text(encoding="utf-8"))
    strict, exploratory, review, rejected = [], [], [], []
    for s in r197.get("candidates", []):
        item = {
            "candidate": f"{s['candidate_id']}_{s['candidate_name']}",
            "zip": s["zip"],
            "dataset2_mad_vs_177b": s["dataset2_mad_vs_177b"],
            "changed_rows_vs_177b": s["changed_rows_vs_177b"],
            "top1_change_vs_177b": s["top1_change_vs_177b"],
            "dataset2_mad_vs_143": s["dataset2_mad_vs_143"],
            "top1_change_vs_143": s["top1_change_vs_143"],
            "submit_score": score(s),
        }
        if strict_ok(s):
            strict.append(item)
        elif exploratory_ok(s):
            exploratory.append(item)
        elif s.get("validate_pass") and s.get("zip_root_pass"):
            review.append(item)
        else:
            rejected.append(item)
    strict.sort(key=lambda x: x["submit_score"], reverse=True)
    exploratory.sort(key=lambda x: x["submit_score"], reverse=True)
    best = strict[0] if strict else (exploratory[0] if exploratory else None)
    backup = (strict[1] if len(strict) > 1 else (exploratory[1] if len(exploratory) > 1 else None))
    out = {
        "id": 198,
        "name": "wsl_refined_candidate_eval",
        "strict_submit_candidates": strict,
        "exploratory_submit_candidates": exploratory,
        "review_only_candidates": review,
        "rejected_candidates": rejected,
        "best_submit_candidate": best,
        "backup_candidate": backup,
        "reason": "selected strict first, exploratory second" if best else "no candidate passed gates",
    }
    write_json(analysis / "198_wsl_refined_candidate_eval.json", out)
    lines = ["# 198_wsl_refined_candidate_eval", "", f"- strict_submit_candidates: {len(strict)}", f"- exploratory_submit_candidates: {len(exploratory)}", f"- best_submit_candidate: {best['candidate'] if best else '-'}", f"- backup_candidate: {backup['candidate'] if backup else '-'}"]
    (analysis / "198_wsl_refined_candidate_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"strict": len(strict), "exploratory": len(exploratory), "best": best["candidate"] if best else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
