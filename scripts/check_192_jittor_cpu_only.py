import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"


def main():
    ANALYSIS.mkdir(exist_ok=True)
    result = {
        "id": 192,
        "name": "jittor_cpu_only_check",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "env": {k: os.environ.get(k, "") for k in ["use_cuda", "nvcc_path", "cache_path"]},
        "jittor_available": False,
        "jittor_geometric_available": False,
        "errors": [],
    }
    try:
        import jittor as jt

        jt.flags.use_cuda = 0
        x = jt.array([1, 2, 3])
        result["jittor_available"] = bool(int(x.sum().data[0]) == 6)
        result["jittor_version"] = jt.__version__
        result["jt_has_cuda"] = str(getattr(jt, "has_cuda", None))
    except Exception as exc:
        result["errors"].append(f"jittor import/run failed: {repr(exc)}")
    try:
        import jittor_geometric

        result["jittor_geometric_available"] = True
        result["jittor_geometric_version"] = getattr(jittor_geometric, "__version__", "unknown")
    except Exception as exc:
        result["errors"].append(f"jittor_geometric import failed: {repr(exc)}")
    (ANALYSIS / "192_jittor_cpu_only_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 192_jittor_cpu_only_check", ""]
    for k in ["jittor_available", "jittor_geometric_available", "jittor_version", "jittor_geometric_version"]:
        if k in result:
            lines.append(f"- {k}: {result[k]}")
    lines.append(f"- env: {result['env']}")
    if result["errors"]:
        lines += ["", "## Errors"] + [f"- {x}" for x in result["errors"]]
    (ANALYSIS / "192_jittor_cpu_only_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ["jittor_available", "jittor_geometric_available", "errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
