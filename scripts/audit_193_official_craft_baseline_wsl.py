import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
EXTRACTED = ROOT / "official_data_recheck" / "extracted_wsl"


def main():
    ANALYSIS.mkdir(exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    for name, dest in [("craft_baseline.zip", "craft_baseline"), ("data_A (1).zip", "data_A")]:
        z = ROOT / "official_data_recheck" / "downloads" / name
        d = EXTRACTED / dest
        d.mkdir(parents=True, exist_ok=True)
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
    files = sorted(str(p.relative_to(ROOT)) for p in EXTRACTED.rglob("*") if p.is_file())
    (ANALYSIS / "193_baseline_files.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    grep_patterns = r"import jittor|jittor as jt|jittor_geometric|jt\.Module|Dataset|train|predict|model|embedding|graph|src|dst|time|c1|c100|candidate|recall|rank"
    grep = subprocess.run(["grep", "-RInE", grep_patterns, str(EXTRACTED)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (ANALYSIS / "193_baseline_grep.txt").write_text(grep.stdout, encoding="utf-8")
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in EXTRACTED.rglob("*.py"))
    out = {
        "id": 193,
        "name": "official_craft_baseline_wsl_audit",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "files": files,
        "baseline_uses_jittor": "import jittor" in text or "jittor as jt" in text,
        "baseline_uses_jittor_geometric": "jittor_geometric" in text,
        "has_train_script": any("train" in p.lower() for p in files) or bool(re.search(r"def\s+train", text)),
        "has_predict_script": any("predict" in p.lower() for p in files) or "predict" in text,
        "has_model_script": any("model" in p.lower() for p in files) or "CRAFT" in text,
        "has_dataset_script": any("dataset" in p.lower() for p in files) or "TemporalData" in text,
        "has_candidate_c1_c100_logic": bool(re.search(r"c1|c100|candidates?", text, re.I)),
        "has_graph_logic": bool(re.search(r"graph|neighbor|TemporalData|CRAFT", text, re.I)),
        "has_time_logic": "time" in text or ".t" in text,
        "has_submission_generation": bool(re.search(r"to_csv|submission|result", text, re.I)),
        "should_return_to_jittor_framework": "true",
    }
    (ANALYSIS / "193_official_craft_baseline_wsl.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 193_official_craft_baseline_wsl", ""]
    for k, v in out.items():
        if k not in {"files", "created_at"}:
            lines.append(f"- {k}: {v}")
    (ANALYSIS / "193_official_craft_baseline_wsl.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k.startswith("baseline_") or k.startswith("has_") or k.startswith("should_")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
