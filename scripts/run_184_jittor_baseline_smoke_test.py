import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
BASELINE = ROOT / "official_data_recheck" / "extracted" / "craft_baseline"


def run(cmd, cwd=ROOT, timeout=60):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "ok": proc.returncode == 0,
    }


def main():
    ANALYSIS.mkdir(exist_ok=True)
    results = []
    results.append(run(["python", "-c", "import sys; print(sys.version)"], timeout=20))
    results.append(run(["python", "-c", "import jittor as jt; print('jittor', jt.__version__)"], timeout=120))
    results.append(run(["python", "-c", "import jittor_geometric; print('jittor_geometric ok')"], timeout=60))
    main_py = BASELINE / "main.py"
    syntax_ok = False
    if main_py.exists():
        syntax = run(["python", "-m", "py_compile", str(main_py)], timeout=60)
        results.append(syntax)
        syntax_ok = syntax["ok"]
    jittor_ok = results[1]["ok"]
    jg_ok = results[2]["ok"]
    smoke_possible = bool(jittor_ok and jg_ok and syntax_ok)
    out = {
        "id": 184,
        "name": "jittor_baseline_smoke_test",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "baseline_dir": str(BASELINE.relative_to(ROOT)) if BASELINE.exists() else str(BASELINE),
        "jittor_available": jittor_ok,
        "jittor_geometric_available": jg_ok,
        "baseline_py_compile_ok": syntax_ok,
        "smoke_possible_without_install": smoke_possible,
        "generated_submission": False,
        "zip": "",
        "auto_eval_decision": "do_not_submit",
        "results": results,
    }
    if not smoke_possible:
        out["reason"] = "Current Python environment cannot run official baseline smoke fully; missing Jittor/JittorGeometric or syntax precheck failed."
    else:
        out["reason"] = "Dependencies import and baseline syntax pass; full training/inference smoke is intentionally not launched here to avoid long GPU/CUDA job."
    (ANALYSIS / "184_jittor_baseline_smoke_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 184_jittor_baseline_smoke_test",
        "",
        f"- jittor_available: {out['jittor_available']}",
        f"- jittor_geometric_available: {out['jittor_geometric_available']}",
        f"- baseline_py_compile_ok: {out['baseline_py_compile_ok']}",
        f"- smoke_possible_without_install: {out['smoke_possible_without_install']}",
        f"- generated_submission: {out['generated_submission']}",
        f"- auto_eval_decision: {out['auto_eval_decision']}",
        f"- reason: {out['reason']}",
        "",
        "184 只做环境与 baseline 可运行性 smoke，不生成自动提交包。",
    ]
    (ANALYSIS / "184_jittor_baseline_smoke_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"smoke_possible_without_install": smoke_possible, "jittor": jittor_ok, "jittor_geometric": jg_ok}, ensure_ascii=False))


if __name__ == "__main__":
    main()
