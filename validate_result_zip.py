from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path


EXPECTED = {"dataset1.csv": 61051, "dataset2.csv": 153420}


def validate_member(zf: zipfile.ZipFile, name: str, expected_rows: int) -> tuple[int, int]:
    rows = 0
    width = None
    with zf.open(name) as raw:
        text = (line.decode("utf-8").rstrip("\n\r") for line in raw)
        reader = csv.reader(text)
        for row in reader:
            if width is None:
                width = len(row)
                if width != 100:
                    raise ValueError(f"{name}: expected 100 columns, got {width}")
            if len(row) != width:
                raise ValueError(f"{name}: inconsistent column count at row {rows + 1}")
            for value in row:
                if "." not in value or len(value.rsplit(".", 1)[-1]) != 8:
                    raise ValueError(f"{name}: value is not 8-decimal formatted at row {rows + 1}: {value}")
                x = float(value)
                if x < 0.0 or x > 1.0:
                    raise ValueError(f"{name}: value out of [0,1] at row {rows + 1}: {value}")
            rows += 1
    if rows != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} rows, got {rows}")
    return rows, width or 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, default="outputs/website_submission/result.zip")
    args = parser.parse_args()
    zip_path = Path(args.zip)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        if names != set(EXPECTED):
            raise ValueError(f"zip members mismatch: {sorted(names)}")
        result = {name: validate_member(zf, name, rows) for name, rows in EXPECTED.items()}
    print({"zip": str(zip_path), "valid": True, "files": result})


if __name__ == "__main__":
    main()
