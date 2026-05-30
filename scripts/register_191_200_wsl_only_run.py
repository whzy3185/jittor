import csv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
CSV_PATH = ROOT / "experiments.csv"
JSONL_PATH = ROOT / "experiments.jsonl"
REGISTRY_PATH = ROOT / "SUBMISSION_REGISTRY_CN.md"
BEST_PATH = ROOT / "BEST_KNOWN_SUBMISSION_CN.md"
RESULT_JSON = Path("/mnt/e/Jitter/.tmp/submit-agent/submit_track1_199_result.json")
BEST_SCORE = 1.2170530157111252

FIELDS = [
    "package_id",
    "package_name",
    "zip_path",
    "dataset1_source",
    "dataset2_source",
    "input_experts",
    "method_family",
    "generation_command",
    "validation_command",
    "format_check_status",
    "top1_change_vs_121",
    "d1_mad_vs_121",
    "d2_mad_vs_121",
    "local_validation_score_if_any",
    "online_score",
    "online_status",
    "risk_level",
    "submit_priority",
    "notes",
    "created_at",
    "submitted_at",
]


def load(rel):
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_200():
    d = load("analysis/200_wsl_final_submit_decision.json")
    r = load(RESULT_JSON)
    latest = r.get("latestRow") or {}
    score_s = latest.get("score")
    try:
        score = float(score_s)
    except (TypeError, ValueError):
        score = None
    status = "online_best" if score is not None and score > BEST_SCORE else ("regressed_or_not_best" if score is not None else "submitted_waiting_score")
    if r:
        d.update(
            {
                "auto_submit_attempted": True,
                "submit_script": "/mnt/e/Jitter/.tmp/submit-agent/submit_track1_199_monitor.js",
                "submit_result_json": str(RESULT_JSON),
                "request_id": latest.get("requestId", ""),
                "submitted_at": latest.get("submittedAt", ""),
                "online_score": score_s,
                "online_status": status,
                "latest_row": latest,
                "submit_result_ok": r.get("ok"),
            }
        )
        (ROOT / "analysis/200_wsl_final_submit_decision.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            "# 200_wsl_final_submit_decision",
            "",
            f"- decision: {d.get('decision')}",
            f"- submit_target: `{d.get('submit_target')}`",
            f"- auto_submit_attempted: {d.get('auto_submit_attempted')}",
            f"- request_id: {d.get('request_id')}",
            f"- submitted_at: {d.get('submitted_at')}",
            f"- online_score: {d.get('online_score')}",
            f"- online_status: {d.get('online_status')}",
            f"- current_best_before_submit: 177b / {BEST_SCORE}",
            "- result: 199 did not beat 177b; current best remains 177b.",
        ]
        (ROOT / "analysis/200_wsl_final_submit_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def record(pid, name, method, command, notes, zip_path="", online_status="not_submitted", online_score="not_submitted", risk="analysis", priority="do_not_submit", fmt="not_applicable", submitted_at="not_submitted"):
    return {
        "package_id": str(pid),
        "package_name": name,
        "zip_path": zip_path,
        "dataset1_source": "177b_w010_top1_guard" if zip_path else "analysis",
        "dataset2_source": "wsl_real_retrieval_refine" if zip_path else "analysis",
        "input_experts": "official_data_recovered,177b,143,retrieval,Jittor_baseline_audit",
        "method_family": method,
        "generation_command": command,
        "validation_command": ("python validate_result_zip.py --zip " + zip_path) if zip_path else "not_applicable",
        "format_check_status": fmt,
        "top1_change_vs_121": "",
        "d1_mad_vs_121": "",
        "d2_mad_vs_121": "",
        "local_validation_score_if_any": "wsl_analysis",
        "online_score": str(online_score),
        "online_status": online_status,
        "risk_level": risk,
        "submit_priority": priority,
        "notes": notes,
        "created_at": NOW,
        "submitted_at": submitted_at,
    }


def build_records(d200):
    r191 = load("analysis/191_wsl_only_environment.json")
    r192 = load("analysis/192_jittor_cpu_only_check.json")
    r193 = load("analysis/193_official_craft_baseline_wsl.json")
    r194 = load("analysis/194_jittor_baseline_smoke_wsl.json")
    r195 = load("analysis/195_baseline_logic_extraction.json")
    r196 = load("analysis/196_177b_success_source_wsl.json")
    r197 = load("analysis/197_wsl_refined_retrieval_grid.json")
    r198 = load("analysis/198_wsl_refined_candidate_eval.json")
    a199 = load("outputs/website_submission_199_final_wsl_candidate/auto_eval.json")
    return [
        record(191, "191_wsl_only_environment", "wsl_env_check", "python scripts/check_191_wsl_only_environment.py", f"pwd_ok={r191.get('summary',{}).get('pwd_ok')}; python_from_wsl_venv={r191.get('summary',{}).get('python_from_wsl_venv')}; validate_177b={r191.get('summary',{}).get('validate_177b_pass')}"),
        record(192, "192_jittor_cpu_only_check", "jittor_cpu_check", "python scripts/check_192_jittor_cpu_only.py", f"jittor={r192.get('jittor_available')}; jittor_geometric={r192.get('jittor_geometric_available')}"),
        record(193, "193_official_craft_baseline_wsl", "baseline_audit", "python scripts/audit_193_official_craft_baseline_wsl.py", f"uses_jittor={r193.get('baseline_uses_jittor')}; uses_jittor_geometric={r193.get('baseline_uses_jittor_geometric')}; candidate_logic={r193.get('has_candidate_c1_c100_logic')}"),
        record(194, "194_jittor_baseline_smoke_wsl", "baseline_smoke", "python scripts/run_194_jittor_baseline_smoke_wsl.py", f"smoke_pass={r194.get('smoke_pass')}; baseline_zip_generated={r194.get('baseline_zip_generated')}; reason={r194.get('failure_reason')}"),
        record(195, "195_baseline_logic_extraction", "baseline_logic", "python scripts/extract_195_baseline_logic.py", f"usable_for_retrieval_refine={r195.get('usable_for_retrieval_refine')}; rules={r195.get('recommended_refine_rules')}"),
        record(196, "196_177b_success_source_wsl", "success_analysis", "python scripts/analyze_196_177b_success_source_wsl.py", f"changed={r196.get('changed_rows_vs_143')}; mad={r196.get('dataset2_mad_vs_143')}; gain={r196.get('gain_source')}"),
        record(197, "197_wsl_refined_retrieval_grid", "wsl_retrieval_grid", "python scripts/make_197_wsl_refined_retrieval_grid.py", f"generated={r197.get('generated_candidates')}; valid={r197.get('valid_candidates')}"),
        record(198, "198_wsl_refined_candidate_eval", "candidate_eval", "python scripts/evaluate_198_wsl_refined_candidates.py", f"strict={len(r198.get('strict_submit_candidates') or [])}; exploratory={len(r198.get('exploratory_submit_candidates') or [])}; best={(r198.get('best_submit_candidate') or {}).get('candidate')}"),
        record(199, "199_final_wsl_candidate", "final_candidate", "python scripts/make_199_final_wsl_candidate.py", f"source={a199.get('source_candidate')}; mad177b={a199.get('dataset2_mad_vs_177b')}; top1_177b={a199.get('top1_change_vs_177b')}; decision={a199.get('auto_eval_decision')}", zip_path="submissions/199_final_wsl_candidate_repack_checked/result.zip", online_status=d200.get("online_status", "not_submitted"), online_score=d200.get("online_score") or "not_submitted", risk="submitted_regressed" if d200.get("online_status") == "regressed_or_not_best" else "submitted", priority=d200.get("decision", "do_not_submit"), fmt="valid", submitted_at=d200.get("submitted_at") or "not_submitted"),
        record(200, "200_wsl_final_submit_decision", "submit_decision", "python scripts/make_200_wsl_final_submit_decision.py", f"decision={d200.get('decision')}; target={d200.get('submit_target')}; request_id={d200.get('request_id')}; status={d200.get('online_status')}", online_status=d200.get("online_status", "not_submitted"), online_score=d200.get("online_score") or "not_submitted", risk=d200.get("decision", "decision"), priority=d200.get("decision", "do_not_submit"), submitted_at=d200.get("submitted_at") or "not_submitted"),
    ]


def upsert_jsonl(records):
    existing = {}
    if JSONL_PATH.exists():
        for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                key = str(item.get("package_id", item.get("id", "")))
                if key:
                    existing[key] = item
    for item in records:
        existing[item["package_id"]] = item
    rows = sorted(existing.values(), key=lambda x: int(str(x.get("package_id", x.get("id", 999999)))))
    JSONL_PATH.write_text("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows) + "\n", encoding="utf-8")


def upsert_csv(records):
    existing = {}
    if CSV_PATH.exists():
        with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("package_id"):
                    existing[row["package_id"]] = row
    for item in records:
        existing[item["package_id"]] = {field: item.get(field, "") for field in FIELDS}
    rows = sorted(existing.values(), key=lambda x: int(str(x.get("package_id", 999999))))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def update_registry(d200):
    marker = "## 191_200_wsl_only_retrieval_refine"
    a199 = load("outputs/website_submission_199_final_wsl_candidate/auto_eval.json")
    r198 = load("analysis/198_wsl_refined_candidate_eval.json")
    section = f"""
{marker}

- 当前 best before run：177b_w010_top1_guard / {BEST_SCORE}
- WSL-only：是，训练/生成/校验均从 WSL venv 执行。
- 197 generated：24
- 197 valid：24
- 198 strict_submit_candidates：{len(r198.get('strict_submit_candidates') or [])}
- 198 exploratory_submit_candidates：{len(r198.get('exploratory_submit_candidates') or [])}
- 198 best_submit_candidate：{(r198.get('best_submit_candidate') or {}).get('candidate')}
- 199 source：{a199.get('source_candidate')}
- 199 zip：`submissions/199_final_wsl_candidate_repack_checked/result.zip`
- 199 dataset2_mad_vs_177b：{a199.get('dataset2_mad_vs_177b')}
- 199 top1_change_vs_177b：{a199.get('top1_change_vs_177b')}
- 200 decision：{d200.get('decision')}
- 200 request_id：{d200.get('request_id')}
- 200 online_score：{d200.get('online_score')}
- 200 online_status：{d200.get('online_status')}
- 结论：199 未超过 177b，当前 best 仍为 177b。w=0.075 方向相对 177b 过度偏离，下一步应围绕更窄/分段的 177b retrieval refinement，而不是全局继续放大。
"""
    text = REGISTRY_PATH.read_text(encoding="utf-8", errors="replace") if REGISTRY_PATH.exists() else ""
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n" + section.strip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + section.strip() + "\n"
    REGISTRY_PATH.write_text(text, encoding="utf-8")
    # Preserve best because 199 did not beat it.
    BEST_PATH.write_text(
        f"""# BEST_KNOWN_SUBMISSION_CN

- 当前已知最好包：177b_w010_top1_guard
- 线上分数：{BEST_SCORE}
- request_id：2026053018275085120483
- zip：`submissions/177b_w010_top1_guard_repack_checked/result.zip`
- base：143_score_shape_rebuild
- 方法：真实数据派生 retrieval，top1 guard，dataset1 复制 143
- 更新时间：{NOW}

199_final_wsl_candidate 已提交但未超过 177b：{d200.get('online_score')}。
""",
        encoding="utf-8",
    )


def main():
    d200 = update_200()
    records = build_records(d200)
    upsert_jsonl(records)
    upsert_csv(records)
    update_registry(d200)
    print(json.dumps({"registered": [r["package_id"] for r in records], "online_status": d200.get("online_status"), "online_score": d200.get("online_score")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
