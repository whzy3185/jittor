from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml"}
SKIP_DIRS = {"outputs", "data", "__pycache__", ".git", ".pytest_cache", ".mypy_cache"}


def _iter_source_files(root: Path):
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES:
            yield path, rel


def make_source_zip(root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, rel in _iter_source_files(root):
            zf.write(path, arcname=str(rel))
    return output_path


def make_submission_bundle(root: Path, output_path: Path, dataset1_zip: Path, dataset2_zip: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset1_zip.exists():
        raise FileNotFoundError(dataset1_zip)
    if not dataset2_zip.exists():
        raise FileNotFoundError(dataset2_zip)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset1_zip": str(dataset1_zip),
        "dataset2_zip": str(dataset2_zip),
        "working_dir": str(root),
        "format": "Each dataset zip contains result.json with sorted candidate ids per query.",
        "leakage_control": "dataset2 heuristic fallback uses split=0 by default; split=1 validation edges are excluded unless explicitly enabled.",
    }
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(dataset1_zip, arcname="dataset1.zip")
        zf.write(dataset2_zip, arcname="dataset2.zip")
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/release")
    parser.add_argument("--dataset1-zip", type=str, default="outputs/track1_heuristic_aggressive/dataset1.zip")
    parser.add_argument("--dataset2-zip", type=str, default="outputs/track1_heuristic_aggressive/dataset2.zip")
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = Path(args.output_dir)
    source_zip = make_source_zip(root, output_dir / "track1_aggressive_source.zip")
    submission_bundle = make_submission_bundle(
        root,
        output_dir / "track1_submission_bundle.zip",
        Path(args.dataset1_zip),
        Path(args.dataset2_zip),
    )
    print(json.dumps({"source_zip": str(source_zip), "submission_bundle": str(submission_bundle)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
