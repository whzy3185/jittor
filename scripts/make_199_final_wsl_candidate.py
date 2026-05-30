import json
import shutil
from pathlib import Path

from wsl_retrieval_refine_utils import ensure_dir, repo_root, write_json


def main():
    root = repo_root()
    analysis = root / "analysis"
    r198 = json.loads((analysis / "198_wsl_refined_candidate_eval.json").read_text(encoding="utf-8"))
    best = r198.get("best_submit_candidate")
    out_dir = root / "outputs" / "website_submission_199_final_wsl_candidate"
    sub_dir = root / "submissions" / "199_final_wsl_candidate_repack_checked"
    if not best:
        auto = {"id": 199, "name": "final_wsl_candidate", "generated": False, "auto_eval_decision": "do_not_submit", "reason": "198 had no best_submit_candidate"}
        ensure_dir(out_dir)
        write_json(out_dir / "auto_eval.json", auto)
        write_json(out_dir / "summary.json", auto)
        print(json.dumps(auto, ensure_ascii=False))
        return
    src_dir = root / "outputs" / ("website_submission_" + best["candidate"])
    src_zip = root / best["zip"]
    ensure_dir(out_dir)
    ensure_dir(sub_dir)
    shutil.copy2(src_dir / "dataset1.csv", out_dir / "dataset1.csv")
    shutil.copy2(src_dir / "dataset2.csv", out_dir / "dataset2.csv")
    shutil.copy2(src_zip, sub_dir / "result.zip")
    summary = json.loads((src_dir / "summary.json").read_text(encoding="utf-8"))
    auto = {**summary, "id": 199, "name": "final_wsl_candidate", "source_candidate": best["candidate"], "generated": True, "zip": "submissions/199_final_wsl_candidate_repack_checked/result.zip"}
    gates = [
        auto["dataset1_mad_vs_177b"] == 0,
        0.001 <= auto["dataset2_mad_vs_177b"] <= 0.04,
        auto["changed_rows_vs_177b"] >= 10000,
        auto["top1_change_vs_177b"] <= 0.02,
        auto["dataset1_mad_vs_143"] == 0,
        0.01 <= auto["dataset2_mad_vs_143"] <= 0.08,
        auto["top1_change_vs_143"] <= 0.02,
        auto["validate_pass"],
        auto["zip_root_pass"],
        auto["real_data_derived_signal"],
        auto["wsl_generated"],
        auto["not_windows_python_generated"],
        auto["not_raw_168"],
        auto["not_lgbm_takeover"],
        auto["not_router_expansion"],
        auto["not_score_shape_only"],
    ]
    auto["auto_eval_decision"] = "submit" if all(gates) else "do_not_submit"
    write_json(out_dir / "summary.json", auto)
    write_json(out_dir / "auto_eval.json", auto)
    (root / "LATEST_MAJOR_199_CN.md").write_text(f"# LATEST_MAJOR_199_CN\n\n- source_candidate: {best['candidate']}\n- zip: `{auto['zip']}`\n- auto_eval_decision: {auto['auto_eval_decision']}\n- dataset2_mad_vs_177b: {auto['dataset2_mad_vs_177b']}\n- top1_change_vs_177b: {auto['top1_change_vs_177b']}\n", encoding="utf-8")
    print(json.dumps({"generated": True, "source": best["candidate"], "decision": auto["auto_eval_decision"], "zip": auto["zip"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
