from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path("outputs/smoke_data")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    rows = []
    t = 0
    for u in range(1, 9):
        for _ in range(8):
            rows.append({"src": u, "dst": int(rng.randint(20, 35)), "time": t, "split": "train"})
            t += 1
    for i in range(12):
        rows.append({"src": int(rng.randint(1, 9)), "dst": int(rng.randint(20, 35)), "time": t, "split": "valid"})
        t += 1
    pd.DataFrame(rows).to_csv(root / "edges.csv", index=False)
    test_rows = []
    for q in range(10):
        src = int(rng.randint(1, 9))
        time = t + q
        for j in range(6):
            test_rows.append({"candidate_id": f"q{q}_{j}", "query_id": f"q{q}", "src": src, "dst": int(rng.randint(20, 35)), "time": time})
    pd.DataFrame(test_rows).to_csv(root / "test_candidates.csv", index=False)

    subprocess.check_call([sys.executable, "train.py", "--data-dir", str(root), "--output-dir", "outputs/smoke", "--epochs", "1", "--batch-size", "16", "--emb-dim", "16", "--hidden-dim", "32", "--time-dim", "8", "--negatives", "2", "--val-negatives", "4"])
    subprocess.check_call([sys.executable, "infer.py", "--data-dir", str(root), "--checkpoint", "outputs/smoke/checkpoints/best.pkl", "--meta", "outputs/smoke/checkpoints/best_meta.json", "--output-dir", "outputs/smoke/submission"])
    result = json.loads((Path("outputs/smoke/submission/result.json")).read_text(encoding="utf-8"))
    assert len(result) == len(test_rows)
    print(f"smoke ok: {len(result)} predictions")


if __name__ == "__main__":
    main()
