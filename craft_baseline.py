"""Preserved official CRAFT baseline entry.

This file intentionally delegates to the vendored JittorGeometric CRAFT example
instead of modifying the baseline in place. Use it as a reference run when the
official TGB/JODIE-style data is available.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    baseline = Path(__file__).resolve().parent / "third_party" / "JittorGeometric" / "examples" / "craft_example.py"
    runpy.run_path(str(baseline), run_name="__main__")
