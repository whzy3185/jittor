import json
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
SRC = ROOT / "official_data_recheck" / "extracted_wsl" / "craft_baseline"
WORK = ROOT / "workspace_baseline_wsl"


def run(cmd, cwd=ROOT, timeout=120):
    p = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True, timeout=timeout)
    return {"cmd": cmd, "ok": p.returncode == 0, "returncode": p.returncode, "stdout": p.stdout[-3000:], "stderr": p.stderr[-3000:]}


def main():
    ANALYSIS.mkdir(exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    if SRC.exists():
        shutil.copytree(SRC, WORK)
    main_py = WORK / "main.py"
    patched = False
    if main_py.exists():
        s = main_py.read_text(encoding="utf-8", errors="replace")
        s = s.replace("jt.flags.use_cuda = 1", "jt.flags.use_cuda = 0")
        main_py.write_text(s, encoding="utf-8")
        patched = True
    results = {
        "py_compile": run("python -m py_compile workspace_baseline_wsl/main.py", timeout=120),
        "imports": run("python - <<'PY'\nimport jittor as jt\njt.flags.use_cuda=0\nfrom jittor_geometric.data import TemporalData\nfrom jittor_geometric.nn.models.craft import CRAFT\nprint('imports ok')\nPY", timeout=240),
        "data_read": run("python - <<'PY'\nimport pandas as pd\nbase='official_data_recheck/extracted_wsl/data_A/data_A (1)'\nfor ds in ['dataset1','dataset2']:\n    tr=pd.read_csv(f'{base}/{ds}/train.csv', nrows=5)\n    te=pd.read_csv(f'{base}/{ds}/test.csv', nrows=5)\n    print(ds, tr.shape, te.shape, list(te.columns[:5]))\nPY", timeout=120),
    }
    smoke_pass = all(x["ok"] for x in results.values())
    out = {
        "id": 194,
        "name": "jittor_baseline_smoke_wsl",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "workspace_baseline": str(WORK.relative_to(ROOT)),
        "patched_use_cuda_0": patched,
        "executed": True,
        "smoke_pass": smoke_pass,
        "baseline_zip_generated": False,
        "validate_pass": False,
        "failure_reason": "" if smoke_pass else "baseline import/compile/data-read smoke did not fully pass; see results",
        "results": results,
        "auto_eval_decision": "do_not_submit",
    }
    (ANALYSIS / "194_jittor_baseline_smoke_wsl.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 194_jittor_baseline_smoke_wsl", "", f"- executed: {out['executed']}", f"- smoke_pass: {out['smoke_pass']}", f"- baseline_zip_generated: {out['baseline_zip_generated']}", f"- validate: {'PASS' if out['validate_pass'] else 'not_generated'}", f"- failure_reason: {out['failure_reason'] or '-'}"]
    (ANALYSIS / "194_jittor_baseline_smoke_wsl.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["executed", "smoke_pass", "baseline_zip_generated", "failure_reason"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
