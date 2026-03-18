"""
clean.py
==============
Step 2 (Part A): Clean raw TXT files produced by step1_collect.py.

WHY THIS STEP EXISTS:
  The TXT files from Step 1 were extracted from raw iXBRL HTML. They still
  contain viewer noise (JavaScript warnings, XBRL boilerplate) and messy
  whitespace. This script produces clean, normalized text that Step 2 (Part B)
  — the shadow ingestor — can reliably split into Section and Shadow Nodes.

WHAT IT DOES:
  1. Strips viewer/XBRL noise lines (e.g. "Please enable JavaScript...")
  2. Fast-forwards to the real document start (the SEC header line)
  3. Normalizes Item headers so they're consistently detectable
     e.g. "ITEM 1A." and "Item 1A." both become "Item 1A."
  4. Collapses excess whitespace

INPUT:
  sec_txt/annual/      → e.g. TSLA_10K_2022.txt
  sec_txt/quarterly/   → e.g. TSLA_10Q_2024-10-24.txt

OUTPUT:
  sec_txt_clean/annual/      → e.g. TSLA_10K_2022.clean.txt
  sec_txt_clean/quarterly/   → e.g. TSLA_10Q_2024-10-24.clean.txt

Usage:
  cd scripts/
  python clean.py           # batch-clean everything
  python clean.py --file ../sec_txt/annual/TSLA_10K_2024.txt
"""

import os
import re
import argparse

# ---------------------------------------------------------------------------
# Directories  (relative to scripts/)
# ---------------------------------------------------------------------------

# Base directory = project root, resolved from this file's location.
# Works regardless of which directory you run the script from.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_DIRS = {
    "annual":    os.path.join(_ROOT, "data", "raw", "sec_txt", "annual"),
    "quarterly": os.path.join(_ROOT, "data", "raw", "sec_txt", "quarterly"),
    "def14a":    os.path.join(_ROOT, "data", "raw", "sec_txt", "def14a"),
    "8k":        os.path.join(_ROOT, "data", "raw", "sec_txt", "8k"),
}

OUTPUT_DIRS = {
    "annual":    os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "annual"),
    "quarterly": os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "quarterly"),
    "def14a":    os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "def14a"),
    "8k":        os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "8k"),
}

# ---------------------------------------------------------------------------
# Noise patterns to strip line-by-line
# These are artifacts from the iXBRL viewer that slip through html_to_text().
# ---------------------------------------------------------------------------

NOISE_PATTERNS = [
    r"^XBRL\s+Viewer\s*$",
    r"^Please enable JavaScript to use the EDGAR Inline XBRL Viewer\.",
    r"^This page uses Javascript\.",
    r"^Your browser either doesn't support Javascript or you have it turned off\.",
    r"^Loading\.\.\.",
    r"^\s*ix:[\w]+",          # leftover iXBRL tag names
    r"^\s*contextRef=",       # leftover XBRL attributes
]

# ---------------------------------------------------------------------------
# Anchors that mark where the real document content begins.
# Everything before the first match is preamble/viewer noise.
# ---------------------------------------------------------------------------

START_ANCHORS = [
    r"UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION",
    r"\bFORM\s+10[\-\u2011\u2013]?\s*[KQ]\b",   # FORM 10-K or FORM 10-Q
    r"\bANNUAL\s+REPORT\b",
    r"\bQUARTERLY\s+REPORT\b",
]

# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def strip_noise_lines(text: str) -> str:
    """
    Remove lines that are pure viewer/XBRL noise.
    We check each line against NOISE_PATTERNS and drop matches.
    """
    lines = text.splitlines()
    clean = []
    for line in lines:
        stripped = line.strip()
        if any(re.search(pat, stripped, re.IGNORECASE) for pat in NOISE_PATTERNS):
            continue  # drop the noise line
        clean.append(line)
    return "\n".join(clean)


def find_document_start(text: str) -> int:
    """
    Scan for the first START_ANCHOR match and return its character position.
    Everything before this is pre-document boilerplate.
    If nothing matches, return 0 (keep the whole text).
    """
    best = len(text)  # default: keep everything
    for pattern in START_ANCHORS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            best = min(best, match.start())
    return 0 if best == len(text) else best


def normalize_items(text: str) -> str:
    """
    Standardize SEC Item header formatting so the Step 2 section splitter
    can reliably detect them with a single regex.

    Before: "ITEM 1A. RISK FACTORS" or "item 1a risk factors"
    After:  "Item 1A. Risk Factors"  (title-cased, dot after number)

    Also ensures each Item header is on its own line with a blank line before it,
    which is what detect_sections() in step2_ingest.py expects.
    """
    # Normalize "ITEM 1A." / "item 1a." → "Item 1A."
    text = re.sub(
        r"(?m)^[ \t]*(ITEM|item)\s+(\d+[A-Za-z]?)\.?\s+",
        lambda m: f"\nItem {m.group(2).upper()}. ",
        text
    )
    return text


def normalize_whitespace(text: str) -> str:
    """
    - Strip trailing spaces on each line
    - Collapse 3+ consecutive blank lines → 1 blank line
    - Remove [TABLE] / [/TABLE] markers left by html_to_text()
      (we keep the pipe-delimited content but drop the markers)
    """
    text = re.sub(r"\[/?TABLE\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)  # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                   # excess blank lines
    return text.strip()


def clean_text(raw: str) -> str:
    """Full cleaning pipeline for a single document."""
    text = strip_noise_lines(raw)
    start = find_document_start(text)
    text = text[start:]            # drop preamble
    text = normalize_items(text)
    text = normalize_whitespace(text)
    return text

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---------------------------------------------------------------------------
# Single-file and batch modes
# ---------------------------------------------------------------------------

def clean_file(inp: str, out: str):
    """Clean one TXT file and write to out."""
    if os.path.exists(out):
        print(f"   [SKIP] already exists: {out}")
        return
    raw = read(inp)
    cleaned = clean_text(raw)
    write(out, cleaned)
    size_kb = len(cleaned) // 1024
    print(f"   cleaned → {out}  ({size_kb} KB)")


def batch_clean():
    """
    Scan both annual and quarterly input dirs for .txt files and clean them.
    Output filename: same name but with .clean.txt extension.
    """
    for period, in_dir in INPUT_DIRS.items():
        out_dir = OUTPUT_DIRS[period]
        if not os.path.exists(in_dir):
            print(f"[SKIP] directory not found: {in_dir}")
            continue

        txt_files = [f for f in os.listdir(in_dir) if f.endswith(".txt")]
        print(f"\n=== {period.upper()} ({len(txt_files)} files) ===")

        for fname in sorted(txt_files):
            inp = os.path.join(in_dir, fname)
            # e.g. TSLA_10K_2022.txt → TSLA_10K_2022.clean.txt
            out_fname = fname.replace(".txt", ".clean.txt")
            out = os.path.join(out_dir, out_fname)
            clean_file(inp, out)


def main():
    parser = argparse.ArgumentParser(description="Step 2A: Clean raw SEC TXT files")
    parser.add_argument("--file", help="Clean a single file (provide full path)")
    parser.add_argument("--out",  help="Output path (required with --file)")
    args = parser.parse_args()

    if args.file:
        if not args.out:
            # Auto-generate output path: same filename with .clean.txt
            base = args.file.replace(".txt", ".clean.txt")
            args.out = base
        clean_file(args.file, args.out)
    else:
        batch_clean()

    print("\nStep 2A complete.")


if __name__ == "__main__":
    main()
