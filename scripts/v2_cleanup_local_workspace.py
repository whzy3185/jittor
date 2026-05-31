import argparse
from v2_pipeline import cleanup_workspace

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    cleanup_workspace(dry_run=not a.apply)
