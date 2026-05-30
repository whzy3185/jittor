import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"


def run(cmd, timeout=180):
    proc = subprocess.run(cmd, shell=True, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    return {"cmd": cmd, "ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout[-3000:], "stderr": proc.stderr[-3000:]}


def main():
    ANALYSIS.mkdir(exist_ok=True)
    out = {
        "id": 191,
        "name": "wsl_only_environment",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "pwd": str(ROOT),
        "python_executable": sys.executable,
        "env": {k: os.environ.get(k, "") for k in ["nvcc_path", "cache_path", "use_cuda"]},
        "checks": {
            "system": run("pwd; which python; python --version; uname -a; gcc --version | head -1 || true; g++ --version | head -1 || true"),
            "validate_177b": run("python validate_result_zip.py --zip submissions/177b_w010_top1_guard_repack_checked/result.zip", 240),
            "validate_143": run("python validate_result_zip.py --zip submissions/143_score_shape_rebuild_repack_checked/result.zip", 240),
        },
    }
    out["summary"] = {
        "wsl_generated": str(ROOT).startswith("/mnt/e/"),
        "pwd_ok": str(ROOT) == "/mnt/e/Jitter/track1_aggressive_wsl",
        "python_from_wsl_venv": sys.executable.startswith("/mnt/e/Jitter/.venv_wsl_cpu"),
        "use_cuda_is_0": os.environ.get("use_cuda") == "0",
        "cache_path_ok": os.environ.get("cache_path") == "/mnt/e/Jitter/.jittor_wsl_cpu_cache_cpuonly",
        "validate_177b_pass": out["checks"]["validate_177b"]["ok"],
        "validate_143_pass": out["checks"]["validate_143"]["ok"],
    }
    (ANALYSIS / "191_wsl_only_environment.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 191_wsl_only_environment", ""]
    md += [f"- {k}: {v}" for k, v in out["summary"].items()]
    (ANALYSIS / "191_wsl_only_environment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
