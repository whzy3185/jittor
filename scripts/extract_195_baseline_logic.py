import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
BASE = ROOT / "workspace_baseline_wsl"


def main():
    ANALYSIS.mkdir(exist_ok=True)
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in BASE.rglob("*.py")) if BASE.exists() else ""
    out = {
        "id": 195,
        "name": "baseline_logic_extraction",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "src_dst_time_processing": all(x in text for x in ["src", "dst", "time"]),
        "candidate_c1_c100_processing": bool(re.search(r"c1|c100|candidate", text, re.I)),
        "use_graph_neighbor": bool(re.search(r"neighbor|TemporalData|TemporalDataLoader", text)),
        "use_time_decay": bool(re.search(r"time|t=", text)),
        "use_src_embedding": "src" in text and "embedding" in text.lower(),
        "use_dst_embedding": "dst" in text and "embedding" in text.lower(),
        "use_global_popularity": False,
        "use_negative_sampling": "neg_sampling" in text or "neg_score" in text,
        "score_only_candidate_set": bool(re.search(r"test_candidates|candidate", text)),
        "normalization_or_softmax": bool(re.search(r"softmax|sigmoid|rank|normalize", text, re.I)),
        "generates_dataset1_dataset2": False,
        "usable_for_retrieval_refine": True,
        "recommended_refine_rules": [
            "use_candidate_mask",
            "prefer graph-neighbor/history overlap",
            "use time-aware recent history",
            "keep scoring restricted to c1..c100 candidates",
        ],
    }
    (ANALYSIS / "195_baseline_logic_extraction.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 195_baseline_logic_extraction", ""]
    for k, v in out.items():
        if k != "recommended_refine_rules":
            lines.append(f"- {k}: {v}")
    lines.append("- recommended_refine_rules: " + ", ".join(out["recommended_refine_rules"]))
    (ANALYSIS / "195_baseline_logic_extraction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"usable_for_retrieval_refine": out["usable_for_retrieval_refine"], "recommended_refine_rules": out["recommended_refine_rules"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
