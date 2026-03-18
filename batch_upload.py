"""
batch_upload.py
===============
Upload multiple documents from a CSV manifest in one command.

CSV format (with header row):
  file,linked_to,title,ticker,form_type,period

  file       — path to .txt file (relative to this script or absolute)
  linked_to  — node ID to attach the document to via [:LINKED_TO]
  title      — human-readable title for the Document node
  ticker     — stock ticker (e.g. TSLA); leave blank for non-company docs
  form_type  — document type (e.g. article, transcript, note); leave blank if unknown
  period     — period covered (e.g. 2023, 2023-Q4); leave blank if unknown

Usage:
  python batch_upload.py demo_docs/manifest.csv

Output:
  Prints per-doc results. Skips rows where the file does not exist.
  Prints a summary at the end: N uploaded, N skipped, N failed.

Requirements:
  Same as upload.py — pip install openai neo4j python-dotenv pyyaml tiktoken pandas pyarrow scikit-learn numpy
"""

import csv
import os
import sys
import traceback

from upload import upload

_ROOT = os.path.abspath(os.path.dirname(__file__))


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_upload.py <manifest.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"ERROR: manifest not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Manifest: {csv_path}")
    print(f"Rows:     {len(rows)}\n")

    uploaded = 0
    skipped  = 0
    failed   = 0

    for i, row in enumerate(rows, 1):
        file_path = row.get("file", "").strip()
        linked_to = row.get("linked_to", "").strip()
        title     = row.get("title", "").strip()
        ticker    = row.get("ticker", "").strip()
        form_type = row.get("form_type", "").strip()
        period    = row.get("period", "").strip()

        print(f"[{i}/{len(rows)}] {title or file_path}")

        # Resolve relative paths from the script root
        if not os.path.isabs(file_path):
            file_path = os.path.join(_ROOT, file_path)

        if not os.path.exists(file_path):
            print(f"  [SKIP] file not found: {file_path}\n")
            skipped += 1
            continue

        if not linked_to or not title:
            print(f"  [SKIP] missing linked_to or title\n")
            skipped += 1
            continue

        try:
            upload(
                file_path = file_path,
                parent_id = linked_to,
                title     = title,
                ticker    = ticker,
                form_type = form_type,
                period    = period,
            )
            uploaded += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done.  Uploaded: {uploaded}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    main()
