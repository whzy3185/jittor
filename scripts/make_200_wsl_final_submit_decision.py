import json
import subprocess
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from wsl_retrieval_refine_utils import repo_root, write_json


def validate(root: Path, zip_rel: str):
    proc = subprocess.run(["python", "validate_result_zip.py", "--zip", zip_rel], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode == 0, proc.stdout.strip()


def root_ok(root: Path, zip_rel: str):
    with zipfile.ZipFile(root / zip_rel) as zf:
        names = sorted(zf.namelist())
    return names == ["dataset1.csv", "dataset2.csv"], names


def main():
    root = repo_root()
    analysis = root / "analysis"
    auto_path = root / "outputs" / "website_submission_199_final_wsl_candidate" / "auto_eval.json"
    auto = json.loads(auto_path.read_text(encoding="utf-8")) if auto_path.exists() else {}
    decision = "stop_wsl_refined_retrieval_line"
    target = ""
    reasons = []
    if auto.get("generated") and auto.get("auto_eval_decision") == "submit":
        target = auto["zip"]
        val, val_out = validate(root, target)
        zr, names = root_ok(root, target)
        checks = [
            ("not duplicate 177b", "177b_w010_top1_guard" not in target),
            ("not raw 168", auto.get("not_raw_168")),
            ("not lgbm", auto.get("not_lgbm_takeover")),
            ("not router", auto.get("not_router_expansion")),
            ("not score shape only", auto.get("not_score_shape_only")),
            ("validate", val),
            ("zip root", zr),
            ("dataset1", auto.get("dataset1_mad_vs_177b") == 0),
            ("top1", auto.get("top1_change_vs_177b", 1) <= 0.02),
            ("wsl", auto.get("wsl_generated")),
        ]
        reasons = [name for name, ok in checks if not ok]
        decision = "submit_199" if not reasons else "manual_review_required"
    else:
        val_out = ""
        names = []
        reasons.append(auto.get("reason") or "199 not generated or not submittable")
    out = {
        "id": 200,
        "name": "wsl_final_submit_decision",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "decision": decision,
        "submit_target": target if decision == "submit_199" else "",
        "current_best_before_submit": {"name": "177b_w010_top1_guard", "online_score": 1.2170530157111252},
        "auto_submit_attempted": False,
        "online_status": "pending_submit" if decision == "submit_199" else "not_submitted",
        "online_score": None,
        "request_id": "",
        "reasons": reasons,
        "zip_names": names,
        "validate_output": val_out,
    }
    write_json(analysis / "200_wsl_final_submit_decision.json", out)
    lines = ["# 200_wsl_final_submit_decision", "", f"- decision: {decision}", f"- submit_target: `{out['submit_target'] or '-'}`", f"- auto_submit_attempted: {out['auto_submit_attempted']}", f"- reasons: {reasons if reasons else 'all gates passed'}"]
    (analysis / "200_wsl_final_submit_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "target": out["submit_target"], "reasons": reasons}, ensure_ascii=False))


if __name__ == "__main__":
    main()
