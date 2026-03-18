"""
ingest_sec.py
===================
Step 3 (SEC): Build the graph manifest from cleaned SEC filing text.

WHY A MANIFEST (not writing directly to Neo4j):
  Ingest is fast and offline — no API calls, re-runnable any time.
  Summarization is slow and costs money (~776 LLM calls for 16 filings).
  If summarize.py crashes at call 400, we need to resume from 400, not redo 1-400.
  The manifest is that checkpoint: summaries already written survive a crash or re-run
  because ingest.py carries them over by node ID before overwriting the file.
  Without the manifest, every re-run would cost money and take hours.
  (User-upload flow skips the manifest — one doc, ~20 calls, just retry on failure.)

WHY THIS SCRIPT IS THE CORE OF THE PROJECT:
  This is where raw text becomes a knowledge graph. Every other step either
  feeds into this one (steps 1-2) or consumes its output (steps 4-5).
  The JSON manifest this produces IS the graph — nodes, relationships, and
  the text that will be embedded and stored in Neo4j.

WHAT IT PRODUCES (the graph structure):
  Document Node
    └── Section Node  (one per "Item X" header found in the filing)
          └── Shadow Node 0  (first ≤1,000-token chunk of the section)
          └── Shadow Node 1  (next chunk, overlaps 200 tokens with chunk 0)
          └── Shadow Node 2  ...
          [Shadow nodes also linked: chunk_0 →[NEXT_CHUNK]→ chunk_1 → ...]

WHY THIS HIERARCHY:
  - Document Node → anchors everything to a specific filing (e.g. TSLA 10-K 2023)
  - Section Node  → isolates meaningful units: Item 1A = Risk Factors always,
                    Item 7 = MD&A always. Enables "compare Tesla and Ford risk
                    factors" without touching unrelated sections.
  - Shadow Node   → the actual text that gets embedded. Small enough for the
                    embedding model, large enough to contain a full thought.
  - NEXT_CHUNK    → sequential chain for chain-of-thought retrieval: if chunk 5
                    is relevant you can also pull chunks 4 and 6 for context.

SEC-SPECIFIC LOGIC (what this script adds on top of ingest_utils):
  - parse_filename()       → extracts ticker, form type, fiscal year from filename
  - split_into_sections()  → splits on "Item X." headers (SEC mandated structure)

SHARED LOGIC (imported from ingest_utils.py):
  - chunk_section()   → paragraph-first, token-cap fallback chunking
  - count_tokens()    → accurate token counting via tiktoken
  - write_manifest()  → serialize manifest to JSON
  - print_summary()   → print node/rel counts

INPUT:
  sec_txt_clean/annual/      → e.g. TSLA_10K_2023.clean.txt
  sec_txt_clean/quarterly/   → e.g. TSLA_10Q_2024-10-24.clean.txt

OUTPUT:
  manifest/manifest.json     → nodes + relationships for all SEC filings

Usage:
  cd scripts/
  python ingest_sec.py

Requirements:
  pip install tiktoken
"""

import os
import re
import sys
from datetime import datetime, timezone

# Add pipelines/shared/ to path so we can import shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from utils import (
    chunk_section,
    count_tokens,
    empty_manifest,
    append_nodes,
    append_relationships,
    write_manifest,
    print_summary,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base directory = project root, resolved from this file's location
# This works regardless of which directory you run the script from.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_DIRS = {
    "annual":    os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "annual"),
    "quarterly": os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "quarterly"),
    "def14a":    os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "def14a"),
    "8k":        os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "8k"),
}

OUTPUT_PATH = os.path.join(_ROOT, "manifest", "manifest.json")

# Regex to detect SEC "Item X." headers — e.g. "Item 1.", "Item 1A.", "Item 7A."
# step2_clean.py normalizes all Item headers to this consistent format.
ITEM_HEADER_RE = re.compile(r"^Item\s+\d+[A-Z]?\.", re.IGNORECASE | re.MULTILINE)

# ---------------------------------------------------------------------------
# SEC-specific: filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> dict:
    """
    Extract metadata from a cleaned SEC filing filename.

    Format (set by step1_collect.py):
      TSLA_10K_2023.clean.txt       → annual 10-K for fiscal year 2023
      TSLA_10Q_2024-10-24.clean.txt → quarterly 10-Q filed 2024-10-24

    Returns: {ticker, form_type, period, doc_id}
    doc_id is the unique Neo4j node ID for this Document Node.
    """
    base  = filename.replace(".clean.txt", "")   # e.g. "TSLA_10K_2023"
    parts = base.split("_", 2)                   # ["TSLA", "10K", "2023"]

    if len(parts) < 3:
        return {"ticker": parts[0], "form_type": parts[1] if len(parts) > 1 else "?",
                "period": "unknown", "doc_id": base}

    ticker    = parts[0]   # "TSLA"
    form_type = parts[1]   # "10K", "10Q", "DEF14A", "8K"
    period    = parts[2]   # "2023" or "2024-10-24"

    form_display = (
        form_type
        .replace("10K", "10-K")
        .replace("10Q", "10-Q")
        .replace("DEF14A", "DEF 14A")
        .replace("8K", "8-K")
    )

    return {
        "ticker":    ticker,
        "form_type": form_display,
        "period":    period,
        "doc_id":    base,    # e.g. "TSLA_10K_2023"
    }

# ---------------------------------------------------------------------------
# SEC-specific: section splitting on Item headers
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> list[dict]:
    """
    Split a cleaned SEC filing into sections by detecting "Item X." headers.

    WHY ITEM HEADERS:
      SEC filings have a legally mandated structure — Item 1A is always Risk
      Factors, Item 7 is always MD&A, etc. Splitting here means every Shadow
      Node inherits a meaningful label. When you later ask "compare Tesla and
      Ford risk factors", the graph can filter to Item 1A sections instantly
      instead of doing a full document scan.

    FALLBACK:
      If no Item headers are found (malformed file or non-standard formatting),
      the whole document is returned as one section. Chunking still happens,
      just without section-level metadata.

    Returns: list of {title, text} dicts.
    """
    matches = list(ITEM_HEADER_RE.finditer(text))

    if not matches:
        # No Item headers — treat entire doc as a single section
        return [{"title": "Full Document", "text": text.strip()}]

    sections = []
    for i, match in enumerate(matches):
        # Title = the full line of the Item header
        title_start = match.start()
        title_end   = text.find("\n", title_start)
        title = text[title_start:title_end].strip() if title_end != -1 else text[title_start:].strip()

        # Body = text from after the title line to the next Item header (or end)
        body_start = title_end + 1 if title_end != -1 else title_start + len(title)
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        # Skip TOC stubs and empty sections.
        # SEC filings list every Item header twice: once in the Table of Contents
        # (with only a page number as body) and once at the actual section.
        # The TOC entries slip through the `if body` check because they do have
        # a body — just pipe characters and page numbers (~14 tokens of junk).
        # A real section (Risk Factors, MD&A, etc.) always has hundreds of tokens.
        # 50 tokens is safely above any TOC entry and safely below any real section.
        if body and count_tokens(body) >= 50:
            sections.append({"title": title, "text": body})

    return sections

# ---------------------------------------------------------------------------
# Core processing: one file → nodes + relationships
# ---------------------------------------------------------------------------

def process_file(file_path: str) -> tuple[list, list]:
    """
    Process a single clean SEC filing into graph nodes and relationships.

    Returns (nodes, relationships) — both are lists of dicts ready for the manifest.
    """
    filename = os.path.basename(file_path)
    meta     = parse_filename(filename)

    print(f"\n  Processing: {filename}")

    nodes         = []
    relationships = []

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    now = datetime.now(timezone.utc).isoformat()

    # --- Document Node ---
    # summary/text are left empty here — filled in by summarize.py after ingest.
    # Keeping API calls out of ingest means ingest is fast, offline, and re-runnable
    # without needing an OpenAI key.
    nodes.append({
        "id":          meta["doc_id"],
        "type":        "Document",
        "ticker":      meta["ticker"],
        "form_type":   meta["form_type"],
        "period":      meta["period"],
        "source_file": filename,
        "summary":     None,       # filled in by summarize.py
        "embedding":   None,       # filled in by embed.py
        "updated_at":  now,
    })

    # Company →[HAS_DOCUMENT]→ Document
    # Ticker is the Company node ID (created in main() before any files are processed).
    relationships.append({"from": meta["ticker"], "to": meta["doc_id"], "type": "HAS_DOCUMENT"})

    sections = split_into_sections(text)
    print(f"    Sections: {len(sections)}")

    for sec_idx, section in enumerate(sections):
        # Clean title → safe ID slug (e.g. "Item 1A. Risk Factors" → "Item_1A_Risk_Factors")
        # strip() removes leading/trailing underscores left by dots/spaces at title edges
        title_slug = re.sub(r"[^A-Za-z0-9]+", "_", section["title"])[:50].strip("_")
        # sec_idx appended to guarantee uniqueness when two section titles
        # share the same first 50 characters after slugification.
        section_id = f"{meta['doc_id']}__{sec_idx}__{title_slug}"

        # --- Section Node ---
        # full_text = complete section body before chunking.
        # summarize.py reads this to generate a proper LLM summary.
        # full_text is manifest-only — load_neo4j.py does NOT write it to Neo4j.
        # text/summary are null here and filled in by summarize.py.
        nodes.append({
            "id":            section_id,
            "type":          "Section",
            "doc_id":        meta["doc_id"],
            "title":         section["title"],
            "section_index": sec_idx,
            "full_text":     section["text"], # pre-chunk source text for summarize.py
            "summary":       None,            # filled in by summarize.py
            "embedding":     None,            # filled in by embed.py
            "updated_at":    now,
        })

        # Document →[HAS_SECTION]→ Section
        relationships.append({"from": meta["doc_id"], "to": section_id, "type": "HAS_SECTION"})

        # --- Shadow Nodes ---
        chunks = chunk_section(section["text"])
        print(f"      {section['title'][:55]:55s} → {len(chunks)} chunks")

        prev_id = None
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{section_id}__chunk_{chunk_idx}"

            nodes.append({
                "id":          chunk_id,
                "type":        "Shadow",
                "doc_id":      meta["doc_id"],
                "section_id":  section_id,
                "chunk_index": chunk_idx,
                "text":        chunk_text,
                "token_count": count_tokens(chunk_text),
                "embedding":   None,       # filled in by embed.py
                "updated_at":  now,
            })

            # Section →[HAS_SHADOW]→ Shadow
            relationships.append({"from": section_id, "to": chunk_id, "type": "HAS_SHADOW"})

            # Shadow →[NEXT_CHUNK]→ Shadow  (sequential chain)
            if prev_id:
                relationships.append({"from": prev_id, "to": chunk_id, "type": "NEXT_CHUNK"})

            prev_id = chunk_id

    return nodes, relationships

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Collect all .clean.txt files
    files = []
    for _, dir_path in INPUT_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"[SKIP] not found: {dir_path}")
            continue
        for fname in sorted(os.listdir(dir_path)):
            if fname.endswith(".clean.txt"):
                files.append(os.path.join(dir_path, fname))

    if not files:
        print("No .clean.txt files found. Run clean.py first.")
        return

    print(f"Found {len(files)} files. Building SEC manifest...\n")

    manifest = empty_manifest()

    # --- Company Nodes ---
    # One node per ticker, sitting above all its Document nodes.
    # Created upfront so every Document can link to its Company regardless
    # of the order files are processed.
    # Company ID is just the ticker — stable, human-readable, unique.
    now = datetime.now(timezone.utc).isoformat()
    seen_tickers = set()
    for file_path in files:
        ticker = parse_filename(os.path.basename(file_path))["ticker"]
        if ticker not in seen_tickers:
            append_nodes(manifest, [{"id": ticker, "type": "Company", "ticker": ticker, "updated_at": now}])
            seen_tickers.add(ticker)

    for file_path in files:
        nodes, rels = process_file(file_path)
        append_nodes(manifest, nodes)
        append_relationships(manifest, rels)

    # Carry over summaries from the previous manifest so re-running ingest
    # doesn't discard summaries that summarize.py already paid to generate.
    # Matches by node id — only copies summary/text if they were non-null.
    if os.path.exists(OUTPUT_PATH):
        import json
        with open(OUTPUT_PATH) as f:
            old = json.load(f)
        old_summaries = {
            n["id"]: n for n in old["nodes"]
            if n.get("summary")
        }
        for node in manifest["nodes"]:
            old_node = old_summaries.get(node["id"])
            if old_node and old_node.get("summary"):
                node["summary"] = old_node["summary"]
        carried = len(old_summaries)
        print(f"\nCarried over {carried} existing summaries from previous manifest.")

    write_manifest(manifest, OUTPUT_PATH)
    print_summary(manifest)


if __name__ == "__main__":
    main()
