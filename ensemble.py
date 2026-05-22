from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from track1_dynamic_rec.metrics import rank_normalize
from track1_dynamic_rec.submission import package_submission, write_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/track1/ensemble_submission")
    args = parser.parse_args()
    frames = [pd.read_csv(p) for p in args.predictions]
    weights = args.weights or [1.0] * len(frames)
    base = frames[0].copy()
    score = 0.0
    for frame, w in zip(frames, weights):
        score = score + float(w) * rank_normalize(frame, "score")
    base["score"] = score / sum(weights)
    result = write_submission(base, args.output_dir)
    z = package_submission(args.output_dir, Path(args.output_dir).with_suffix(".zip"))
    print(f"wrote {result}")
    print(f"packaged {z}")


if __name__ == "__main__":
    main()
