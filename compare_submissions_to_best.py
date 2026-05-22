from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_from_zip(path: Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(path) as zf:
        with zf.open(name) as f:
            return pd.read_csv(f, header=None).to_numpy(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    root = Path("submissions")
    base = root / args.base / "result.zip"
    base1 = read_csv_from_zip(base, "dataset1.csv")
    base2 = read_csv_from_zip(base, "dataset2.csv")
    for idx in range(args.start, args.end + 1):
        dirs = sorted(root.glob(f"{idx}_*"))
        if not dirs:
            continue
        path = dirs[0] / "result.zip"
        data1 = read_csv_from_zip(path, "dataset1.csv")
        data2 = read_csv_from_zip(path, "dataset2.csv")
        print(
            f"{dirs[0].name}: "
            f"d1_mad={np.mean(np.abs(data1 - base1)):.8f} "
            f"d2_mad={np.mean(np.abs(data2 - base2)):.8f} "
            f"d1_std={data1.std():.8f} "
            f"d2_std={data2.std():.8f} "
            f"size={path.stat().st_size}"
        )


if __name__ == "__main__":
    main()
