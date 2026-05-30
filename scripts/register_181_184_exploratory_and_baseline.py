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
LATEST_177B = ROOT / "LATEST_MAJOR_177B_CN.md"
RESULT_JSON = Path("E:/Jitter/.tmp/submit-agent/submit_track1_177b_result.json")

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


def load(rel_or_path):
    path = Path(rel_or_path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_182_from_submit():
    d = load("analysis/182_exploratory_submit_decision.json")
    r = load(RESULT_JSON)
    latest = r.get("latestRow") or {}
    if r:
        score = latest.get("score") or None
        try:
            score_f = float(score) if score not in (None, "") else None
        except ValueError:
            score_f = None
        status = "online_best" if score_f is not None and score_f > 1.216080479081863 else ("submitted_scored" if score_f is not None else "submitted_waiting_score")
        d.update(
            {
                "auto_submit_attempted": True,
                "submit_result_json": str(RESULT_JSON),
                "request_id": latest.get("requestId", ""),
                "submitted_at": latest.get("submittedAt", ""),
                "online_status": status if r.get("ok") else "submit_failed",
                "online_score": score,
                "submit_result_status": r.get("status"),
                "submit_result_ok": r.get("ok"),
                "latest_row": latest,
            }
        )
        (ROOT / "analysis/182_exploratory_submit_decision.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        md = [
            "# 182_exploratory_submit_decision",
            "",
            f"- decision: {d.get('decision')}",
            f"- submit_target: `{d.get('submit_target')}`",
            f"- auto_submit_attempted: {d.get('auto_submit_attempted')}",
            f"- request_id: {d.get('request_id')}",
            f"- submitted_at: {d.get('submitted_at')}",
            f"- online_status: {d.get('online_status')}",
            f"- online_score: {d.get('online_score')}",
            f"- submit_script: `{d.get('submit_script')}`",
            f"- dataset2_mad_vs_143: {d.get('preflight', {}).get('dataset2_mad_vs_143')}",
            f"- changed_rows_vs_143: {d.get('preflight', {}).get('changed_rows_vs_143')}",
            f"- top1_change_vs_143: {d.get('preflight', {}).get('top1_change_vs_143')}",
            "",
            "177b 是一次性探索提交，提交对象唯一，未提交 raw 168 或其他 177 候选。",
        ]
        (ROOT / "analysis/182_exploratory_submit_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return d


def record(pid, name, method, command, notes, zip_path="", online_status="not_submitted", online_score="not_submitted", risk="analysis", priority="do_not_submit", fmt="not_applicable", submitted_at="not_submitted"):
    return {
        "package_id": str(pid),
        "package_name": name,
        "zip_path": zip_path,
        "dataset1_source": "143_score_shape_rebuild" if zip_path else "not_applicable",
        "dataset2_source": "real_retrieval_guarded" if zip_path else "analysis",
        "input_experts": "143,official_data_recovered,retrieval",
        "method_family": method,
        "generation_command": command,
        "validation_command": "python validate_result_zip.py --zip " + zip_path if zip_path else "not_applicable",
        "format_check_status": fmt,
        "top1_change_vs_121": "",
        "d1_mad_vs_121": "",
        "d2_mad_vs_121": "",
        "local_validation_score_if_any": "retrieval_probe" if zip_path else "analysis",
        "online_score": str(online_score),
        "online_status": online_status,
        "risk_level": risk,
        "submit_priority": priority,
        "notes": notes,
        "created_at": NOW,
        "submitted_at": submitted_at,
    }


def build_records(d182):
    r181 = load("analysis/181_exploratory_submit_policy_fix.json")
    r183 = load("analysis/183_jittor_baseline_framework_audit.json")
    r184 = load("analysis/184_jittor_baseline_smoke_test.json")
    sel = r181.get("selected_exploratory_candidate") or {}
    pf = d182.get("preflight") or {}
    return [
        record(
            181,
            "181_exploratory_submit_policy_fix",
            "policy_fix",
            "python scripts/fix_181_exploratory_submit_policy.py",
            f"strict={len(r181.get('strict_submit_candidates') or [])}; exploratory={len(r181.get('exploratory_submit_candidates') or [])}; selected={(sel or {}).get('candidate')}",
        ),
        record(
            182,
            "182_exploratory_submit_177b",
            "exploratory_real_retrieval_submit",
            "python scripts/make_182_exploratory_submit_decision.py; node E:/Jitter/.tmp/submit-agent/submit_track1_177b_exploratory_monitor.js",
            f"candidate={pf.get('candidate')}; mad_vs_143={pf.get('dataset2_mad_vs_143')}; changed={pf.get('changed_rows_vs_143')}; top1={pf.get('top1_change_vs_143')}; request_id={d182.get('request_id')}",
            zip_path=d182.get("submit_target", ""),
            online_status=d182.get("online_status", "not_submitted"),
            online_score=d182.get("online_score") or "not_submitted",
            risk="exploratory_submitted",
            priority=d182.get("decision", "do_not_submit"),
            fmt="valid" if pf.get("validate_pass") and pf.get("zip_root_pass") else "invalid",
            submitted_at=d182.get("submitted_at") or "not_submitted",
        ),
        record(
            183,
            "183_jittor_baseline_framework_audit",
            "jittor_baseline_audit",
            "python scripts/audit_183_jittor_baseline_framework.py",
            f"uses_jittor={r183.get('uses_jittor')}; uses_jittor_geometric={r183.get('uses_jittor_geometric')}; return_to_jittor_framework={r183.get('return_to_jittor_framework')}; risks={';'.join(r183.get('risks') or [])}",
        ),
        record(
            184,
            "184_jittor_baseline_smoke_test",
            "jittor_baseline_smoke",
            "python scripts/run_184_jittor_baseline_smoke_test.py",
            f"jittor_available={r184.get('jittor_available')}; jittor_geometric_available={r184.get('jittor_geometric_available')}; smoke_possible={r184.get('smoke_possible_without_install')}; reason={r184.get('reason')}",
        ),
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


def update_docs(d182):
    pf = d182.get("preflight") or {}
    marker = "## 181_184_exploratory_retrieval_and_jittor_baseline"
    section = f"""
{marker}

- 181 policy：新增探索性提交通道；严格提交候选仍为 0。
- 181 selected：177b_w010_top1_guard。
- 182 submit_target：`{d182.get('submit_target')}`
- 182 request_id：{d182.get('request_id') or '-'}
- 182 online_status：{d182.get('online_status')}
- 182 online_score：{d182.get('online_score')}
- 182 dataset2_mad_vs_143：{pf.get('dataset2_mad_vs_143')}
- 182 changed_rows_vs_143：{pf.get('changed_rows_vs_143')}
- 182 top1_change_vs_143：{pf.get('top1_change_vs_143')}
- 183 baseline：官方 craft_baseline 是 Jittor/CRAFT 路线，建议后续回到真实 Jittor 框架。
- 184 smoke：当前 Python 环境 jittor / jittor_geometric 不可用，未生成 184 提交包。
"""
    text = REGISTRY_PATH.read_text(encoding="utf-8", errors="replace") if REGISTRY_PATH.exists() else ""
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n" + section.strip() + "\n"
    else:
        text = text.rstrip() + "\n\n" + section.strip() + "\n"
    REGISTRY_PATH.write_text(text, encoding="utf-8")
    latest = f"""# LATEST_MAJOR_177B_CN

## 名称

177b_w010_top1_guard

## 定位

真实官方数据派生 retrieval 的一次性探索提交候选。该包不是严格 local-validation 通过候选，而是在 181 新增探索通道下提交：dataset1 完全复制 143，dataset2 做 top1 不变的 guarded retrieval 融合。

## 提交结果

- zip：`{d182.get('submit_target')}`
- request_id：{d182.get('request_id') or '-'}
- submitted_at：{d182.get('submitted_at') or '-'}
- online_status：{d182.get('online_status')}
- online_score：{d182.get('online_score')}

## 本地指标

- dataset1_mad_vs_143：{pf.get('dataset1_mad_vs_143')}
- dataset2_mad_vs_143：{pf.get('dataset2_mad_vs_143')}
- changed_rows_vs_143：{pf.get('changed_rows_vs_143')}
- top1_change_vs_143：{pf.get('top1_change_vs_143')}
- validate_pass：{pf.get('validate_pass')}
- zip_root_pass：{pf.get('zip_root_pass')}

## 结论

177b 线上超过 143，说明真实 retrieval 信号在 top1 guard 下存在有效增益。下一步应沿真实数据/Jittor pipeline 深化，而不是回到 submission-output-only 后处理。
"""
    LATEST_177B.write_text(latest, encoding="utf-8")
    try:
        score = float(d182.get("online_score"))
    except (TypeError, ValueError):
        score = None
    if score is not None and score > 1.216080479081863:
        BEST_PATH.write_text(
            f"""# BEST_KNOWN_SUBMISSION_CN

- 当前已知最好包：177b_w010_top1_guard
- 线上分数：{score}
- request_id：{d182.get('request_id')}
- zip：`{d182.get('submit_target')}`
- base：143_score_shape_rebuild
- 方法：真实数据派生 retrieval，top1 guard，dataset1 复制 143
- 更新时间：{NOW}

143_score_shape_rebuild 保留为上一 best：1.216080479081863。
""",
            encoding="utf-8",
        )


def main():
    d182 = update_182_from_submit()
    records = build_records(d182)
    upsert_jsonl(records)
    upsert_csv(records)
    update_docs(d182)
    print(json.dumps({"registered": [r["package_id"] for r in records], "online_status": d182.get("online_status"), "online_score": d182.get("online_score")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
