import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"


def load_json(rel):
    path = ROOT / rel
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_exploratory_submit_candidate(c):
    reasons = []
    if not c.get("real_data_derived_signal"):
        reasons.append("real_data_derived_signal is false")
    if c.get("dataset1_mad_vs_143") != 0:
        reasons.append("dataset1_mad_vs_143 is not 0")
    if not c.get("validate_pass"):
        reasons.append("validate_result_zip did not pass")
    if not c.get("zip_root_pass"):
        reasons.append("zip root did not pass")
    if c.get("zip_names") != ["dataset1.csv", "dataset2.csv"]:
        reasons.append("zip root names are not exactly dataset1.csv/dataset2.csv")
    if c.get("top1_change_vs_143", 1) > 0.01:
        reasons.append("top1_change_vs_143 > 0.01")
    mad = c.get("dataset2_mad_vs_143", 0)
    if not (0.01 <= mad <= 0.04):
        reasons.append("dataset2_mad_vs_143 not in [0.01, 0.04]")
    if c.get("changed_rows_vs_143", 0) < 50000:
        reasons.append("changed_rows_vs_143 < 50000")
    if not c.get("not_lgbm_takeover"):
        reasons.append("candidate is LGBM takeover")
    if not c.get("not_router_expansion"):
        reasons.append("candidate is router expansion")
    if not c.get("not_score_shape_only"):
        reasons.append("candidate is score-shape only")
    if c.get("candidate_id") == "168" or "168_real_retrieval" in str(c.get("zip", "")):
        reasons.append("raw 168 is forbidden")
    return len(reasons) == 0, reasons


def score_candidate(c):
    mad = float(c.get("dataset2_mad_vs_143", 0))
    changed = float(c.get("changed_rows_vs_143", 0))
    top1 = float(c.get("top1_change_vs_143", 0))
    # Prefer a middle-strength real retrieval perturbation with top1 unchanged.
    mad_center_bonus = max(0.0, 1.0 - abs(mad - 0.017) / 0.023)
    changed_bonus = min(changed / 150000.0, 1.0)
    top1_bonus = max(0.0, 1.0 - top1 / 0.01)
    weight_penalty = abs(float(c.get("weight", 0.1)) - 0.1)
    guard_bonus = 0.25 if c.get("guard") == "top1" else 0.0
    return mad_center_bonus * 2.0 + changed_bonus + top1_bonus + guard_bonus - weight_penalty


def main():
    ANALYSIS.mkdir(exist_ok=True)
    r177 = load_json("analysis/177_conservative_retrieval_grid_report.json")
    r178 = load_json("analysis/178_retrieval_candidate_eval.json")
    r180 = load_json("analysis/180_guarded_retrieval_submit_decision.json")
    candidates = r177.get("candidates") or []
    evaluated = []
    for c in candidates:
        ok, reasons = is_exploratory_submit_candidate(c)
        item = {
            "candidate": f"{c.get('candidate_id')}_{c.get('candidate_name')}",
            "candidate_id": c.get("candidate_id"),
            "candidate_name": c.get("candidate_name"),
            "zip": c.get("zip"),
            "dataset1_mad_vs_143": c.get("dataset1_mad_vs_143"),
            "dataset2_mad_vs_143": c.get("dataset2_mad_vs_143"),
            "changed_rows_vs_143": c.get("changed_rows_vs_143"),
            "top1_change_vs_143": c.get("top1_change_vs_143"),
            "validate_pass": c.get("validate_pass"),
            "zip_root_pass": c.get("zip_root_pass"),
            "local_validation_not_bad": c.get("local_validation_not_bad"),
            "strict_submit": c.get("auto_eval_decision") == "submit",
            "exploratory_submit": ok,
            "exploratory_reject_reasons": reasons,
            "exploratory_score": score_candidate(c) if ok else -1,
        }
        evaluated.append(item)
    exploratory = [x for x in evaluated if x["exploratory_submit"]]
    exploratory.sort(key=lambda x: x["exploratory_score"], reverse=True)
    selected = exploratory[0] if exploratory else None
    out = {
        "id": 181,
        "name": "exploratory_submit_policy_fix",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "strict_submit_candidates": [x for x in evaluated if x["strict_submit"]],
        "exploratory_submit_candidates": exploratory,
        "selected_exploratory_candidate": selected,
        "policy": {
            "purpose": "allow one guarded real-retrieval exploratory submit when strict local validation blocks all candidates",
            "still_forbidden": ["raw_168", "lgbm_takeover", "router_expansion", "score_shape_only", "bulk_submit"],
            "requires": [
                "real_data_derived_signal",
                "dataset1_mad_vs_143 == 0",
                "validate_result_zip PASS",
                "zip root == dataset1.csv,dataset2.csv",
                "top1_change_vs_143 <= 0.01",
                "0.01 <= dataset2_mad_vs_143 <= 0.04",
                "changed_rows_vs_143 >= 50000",
            ],
        },
        "prior_178_best_submit_candidate": r178.get("best_submit_candidate"),
        "prior_180_decision": r180.get("decision"),
    }
    (ANALYSIS / "181_exploratory_submit_policy_fix.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 181_exploratory_submit_policy_fix",
        "",
        "## 结论",
        "",
        f"- strict_submit_candidates: {len(out['strict_submit_candidates'])}",
        f"- exploratory_submit_candidates: {len(exploratory)}",
        f"- selected_exploratory_candidate: {selected['candidate'] if selected else '-'}",
        "",
        "## 选择理由",
        "",
    ]
    if selected:
        lines += [
            f"- zip: `{selected['zip']}`",
            f"- dataset2_mad_vs_143: {selected['dataset2_mad_vs_143']}",
            f"- changed_rows_vs_143: {selected['changed_rows_vs_143']}",
            f"- top1_change_vs_143: {selected['top1_change_vs_143']}",
            "- local_validation_not_bad 为 false，因此不是严格提交；本策略只把它标为一次性探索提交候选。",
        ]
    else:
        lines.append("- 没有候选满足探索性提交门槛。")
    lines += [
        "",
        "## 禁止项",
        "",
        "- 不提交 raw 168。",
        "- 不提交 LGBM takeover / router expansion / score-shape-only / 批量包。",
        "- 提交前仍需 182 再次做 zip root 与格式校验，并确认单包脚本。",
    ]
    (ANALYSIS / "181_exploratory_submit_policy_fix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected["candidate"] if selected else None, "exploratory_count": len(exploratory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
