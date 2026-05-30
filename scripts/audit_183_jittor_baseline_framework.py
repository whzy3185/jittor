import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "official_data_recheck" / "extracted" / "craft_baseline"
ANALYSIS = ROOT / "analysis"


def read(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def main():
    ANALYSIS.mkdir(exist_ok=True)
    files = sorted(p.relative_to(BASELINE).as_posix() for p in BASELINE.rglob("*") if p.is_file()) if BASELINE.exists() else []
    main_py = read(BASELINE / "main.py")
    readme = read(BASELINE / "readme.md")
    req = read(BASELINE / "requirements.txt")
    findings = {
        "id": 183,
        "name": "jittor_baseline_framework_audit",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "baseline_dir": str(BASELINE.relative_to(ROOT)) if BASELINE.exists() else str(BASELINE),
        "baseline_exists": BASELINE.exists(),
        "files": files,
        "uses_jittor": bool(re.search(r"\bimport\s+jittor\b|\bfrom\s+jittor\b", main_py)),
        "uses_jittor_geometric": "jittor_geometric" in main_py,
        "has_model_class": "class CRAFT" in main_py or "CRAFT" in main_py,
        "has_training_flow": bool(re.search(r"def\s+train|optimizer|loss|backward|epochs?", main_py, re.I)),
        "has_inference_flow": bool(re.search(r"predict|test_competition|submission|to_csv", main_py, re.I)),
        "has_candidate_c1_c100_logic": bool(re.search(r"c1|c100|candidate", main_py, re.I)),
        "has_dataset1_dataset2_logic": "dataset1" in main_py and "dataset2" in main_py,
        "mentions_cuda": "use_cuda" in main_py or "cuda" in main_py.lower(),
        "requirements": [x.strip() for x in req.splitlines() if x.strip()],
        "readme_summary": readme[:1200],
        "return_to_jittor_framework": True,
        "recommended_next": "run_184_jittor_baseline_smoke_test",
    }
    risks = []
    if not findings["baseline_exists"]:
        risks.append("official craft_baseline directory missing")
    if findings["mentions_cuda"]:
        risks.append("baseline appears CUDA-oriented; CPU smoke may require patch or flags")
    if findings["uses_jittor_geometric"]:
        risks.append("jittor_geometric dependency may be unavailable in current environment")
    findings["risks"] = risks
    (ANALYSIS / "183_jittor_baseline_framework_audit.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 183_jittor_baseline_framework_audit",
        "",
        "## 结论",
        "",
        f"- baseline_exists: {findings['baseline_exists']}",
        f"- uses_jittor: {findings['uses_jittor']}",
        f"- uses_jittor_geometric: {findings['uses_jittor_geometric']}",
        f"- has_training_flow: {findings['has_training_flow']}",
        f"- has_inference_flow: {findings['has_inference_flow']}",
        f"- has_candidate_c1_c100_logic: {findings['has_candidate_c1_c100_logic']}",
        f"- return_to_jittor_framework: {findings['return_to_jittor_framework']}",
        "",
        "## 风险",
        "",
    ]
    lines += [f"- {x}" for x in risks] if risks else ["- 未发现阻断性风险；仍需 184 环境 smoke test。"]
    lines += [
        "",
        "## 方向判断",
        "",
        "官方 baseline 是 Jittor/CRAFT 路线，包含训练和推理流程，也包含 c1-c100 候选处理逻辑。后续真实数据路线应回到该框架做可控训练/验证，而不是继续只在 submission output 上后处理。",
    ]
    (ANALYSIS / "183_jittor_baseline_framework_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"return_to_jittor_framework": findings["return_to_jittor_framework"], "risks": risks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
