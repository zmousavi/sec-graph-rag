"""
run_all.py
==========
Run pipelines by source. Execute from the project root:

  python run_all.py --sec        # run SEC pipeline only
  python run_all.py --news       # run news pipeline only (once built)
  python run_all.py --sec --news # run both

Each pipeline runs 3 steps in order: collect → clean → ingest
"""

import subprocess
import sys
import os


def run(script_path: str):
    """Run a Python script from its own directory and exit if it fails."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print('='*60)
    result = subprocess.run(
        [sys.executable, os.path.basename(script_path)],
        cwd=os.path.join(os.path.dirname(__file__), os.path.dirname(script_path))
    )
    if result.returncode != 0:
        print(f"\n[FAILED] {script_path}")
        sys.exit(result.returncode)


def run_pipeline(source: str):
    """Run all 3 steps for a given source pipeline."""
    run(f"pipelines/{source}/collect.py")
    run(f"pipelines/{source}/clean.py")
    run(f"pipelines/{source}/ingest.py")


def main():
    args = sys.argv[1:]

    if not args:
        print("Usage: python run_all.py --sec | --news | --sec --news")
        sys.exit(1)

    if "--sec" in args:
        print("\n>>> Running SEC pipeline")
        run_pipeline("sec")

    if "--news" in args:
        print("\n>>> Running news pipeline")
        run_pipeline("news")

    print("\n\nAll steps complete.")


if __name__ == "__main__":
    main()
