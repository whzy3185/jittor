from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def make_data(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rng = np.random.RandomState(0)
    train_clean = []
    train_noisy = []
    test_noisy = []
    for i in range(8):
        theta = rng.uniform(0, 2 * np.pi, size=128)
        z = rng.uniform(-0.5, 0.5, size=128)
        r = 0.5 + 0.1 * np.sin(3 * theta)
        clean = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)
        noisy = clean + rng.normal(0, 0.02, clean.shape).astype(np.float32)
        train_clean.append(clean)
        train_noisy.append(noisy)
    for i in range(3):
        theta = rng.uniform(0, 2 * np.pi, size=128)
        z = rng.uniform(-0.5, 0.5, size=128)
        clean = np.stack([0.5 * np.cos(theta), 0.5 * np.sin(theta), z], axis=1).astype(np.float32)
        test_noisy.append(clean + rng.normal(0, 0.02, clean.shape).astype(np.float32))
    np.save(root / "train_noisy.npy", np.stack(train_noisy))
    np.save(root / "train_clean.npy", np.stack(train_clean))
    np.save(root / "test_noisy.npy", np.stack(test_noisy))


def main():
    data = Path(".tmp/track2_smoke_data")
    make_data(data)
    subprocess.check_call([
        sys.executable,
        "train.py",
        "--data-dir",
        str(data),
        "--output-dir",
        "outputs/track2_smoke",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--patch-size",
        "64",
        "--variant",
        "baseline",
        "--hidden-dim",
        "32",
        "--k-values",
        "8",
        "--blocks",
        "1",
        "--stages",
        "1",
        "--max-train-batches",
        "1",
        "--max-val-batches",
        "1",
    ])
    subprocess.check_call([
        sys.executable,
        "infer.py",
        "--data-dir",
        str(data),
        "--checkpoint",
        "outputs/track2_smoke/checkpoints/best.pkl",
        "--output-dir",
        "outputs/track2_smoke/submission",
        "--tta",
        "1",
        "--chunk-size",
        "64",
    ])
    assert Path("outputs/track2_smoke/submission.zip").exists()
    print("track2 smoke test passed")


if __name__ == "__main__":
    main()
