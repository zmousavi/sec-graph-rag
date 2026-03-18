"""
ingest_utils.py
===============
Shared utilities for all step3_ingest_*.py scripts.

WHY THIS MODULE EXISTS:
  Both step3_ingest_sec.py and step3_ingest_news.py produce the same
  graph manifest format (nodes + relationships) and use the same chunking
  logic. Keeping that logic here means a fix or improvement in one place
  applies everywhere — no copy-paste drift.

WHAT LIVES HERE:
  - Tokenizer helpers  (count_tokens, encode, tokens_to_text)
  - chunk_section()    (paragraph-first, token-cap fallback chunking)
  - append_to_manifest() (add nodes/rels to a running manifest dict)
  - write_manifest()   (serialize manifest to JSON)
  - print_summary()    (print node/rel counts after processing)

WHAT DOES NOT LIVE HERE:
  - Section splitting logic  — differs per document type (SEC vs news vs transcript)
  - Filename parsing          — differs per document type
  - Input/output directories  — each script sets its own
"""

import os
import re
import json

import tiktoken  # pip install tiktoken

# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------

# cl100k_base is the tokenizer used by GPT-4, Claude, and most embedding models.
# Using a real tokenizer (not character counting) ensures chunk sizes are accurate
# regardless of whether the text is English prose, financial jargon, or tables.
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Maximum tokens per Shadow Node.
# 1,000 tokens ≈ 750 words ≈ 3-4 paragraphs.
MAX_TOKENS = 1000

# Overlap between consecutive Shadow Nodes (20% of MAX_TOKENS).
# Prevents a fact that lands exactly on a boundary from being lost.
OVERLAP_TOKENS = 200


def count_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(TOKENIZER.encode(text))


def encode(text: str) -> list:
    """Encode a string to a list of token IDs."""
    return list(TOKENIZER.encode(text))


def tokens_to_text(tokens: list) -> str:
    """Decode a list of token IDs back to a string."""
    return TOKENIZER.decode(tokens)


# ---------------------------------------------------------------------------
# Shadow chunking  (shared by all ingestors)
# ---------------------------------------------------------------------------

def chunk_section(section_text: str) -> list[str]:
    """
    Split a block of text into Shadow Chunks using a two-level strategy.

    WHY TWO LEVELS:
      SEC sections, news articles, and transcript speeches all have different
      internal structure, but all share the same problem: some logical units
      (paragraphs, speaker blocks) are short, some are enormous.
      Two levels handle both cases cleanly.

    LEVEL 1 — Paragraph boundaries:
      Split on double newlines. Pack paragraphs into a chunk until we hit
      MAX_TOKENS. This keeps a full risk factor or argument in one chunk,
      which produces better embeddings than cutting mid-thought.

    LEVEL 2 — Token window fallback:
      If a single paragraph exceeds MAX_TOKENS (e.g. a financial table rendered
      as one long line, or a very long speech block), slide a token window across
      it. We use the tokenizer so we never break in the middle of a word.

    OVERLAP:
      The last OVERLAP_TOKENS tokens of each chunk are prepended to the next.
      A sentence that straddles a boundary will appear in both chunks, so
      vector search can retrieve it from either side.

    Returns: list of text strings, one per Shadow Node.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]

    chunks       = []   # completed chunk strings
    current_toks = []   # token IDs accumulating for the current chunk

    for para in paragraphs:
        para_toks = encode(para)

        if len(para_toks) > MAX_TOKENS:
            # --- Paragraph too large: flush current chunk, then slide window ---
            if current_toks:
                chunks.append(tokens_to_text(current_toks))
                current_toks = current_toks[-OVERLAP_TOKENS:]  # carry overlap

            start = 0
            while start < len(para_toks):
                end = min(start + MAX_TOKENS, len(para_toks))
                chunks.append(tokens_to_text(para_toks[start:end]))
                start += MAX_TOKENS - OVERLAP_TOKENS  # advance with overlap

            current_toks = []  # overlap already built into sliding window

        elif len(current_toks) + len(para_toks) > MAX_TOKENS:
            # --- Adding this paragraph would overflow: flush and start fresh ---
            if current_toks:
                chunks.append(tokens_to_text(current_toks))
                current_toks = current_toks[-OVERLAP_TOKENS:] + para_toks
            else:
                current_toks = para_toks

        else:
            # --- Paragraph fits: keep accumulating ---
            current_toks.extend(para_toks)

    if current_toks:
        chunks.append(tokens_to_text(current_toks))

    return chunks


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def empty_manifest() -> dict:
    """Return a fresh manifest dict with empty nodes and relationships lists."""
    return {"nodes": [], "relationships": []}


def append_nodes(manifest: dict, nodes: list):
    """Add a list of node dicts to the manifest."""
    manifest["nodes"].extend(nodes)


def append_relationships(manifest: dict, relationships: list):
    """Add a list of relationship dicts to the manifest."""
    manifest["relationships"].extend(relationships)


def write_manifest(manifest: dict, output_path: str):
    """
    Serialize the manifest to JSON and write it to disk.
    Creates parent directories if they don't exist.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written → {output_path}")


def load_manifest(path: str) -> dict:
    """
    Load an existing manifest from disk.
    Used when appending news/transcript nodes to an existing SEC manifest.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(manifest: dict):
    """Print a summary of node and relationship counts."""
    type_counts = {}
    for node in manifest["nodes"]:
        t = node["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    rel_counts = {}
    for rel in manifest["relationships"]:
        t = rel["type"]
        rel_counts[t] = rel_counts.get(t, 0) + 1

    print(f"\nNode counts:")
    for t, count in type_counts.items():
        print(f"  {t:15s}: {count:,}")
    print(f"\nRelationship counts:")
    for t, count in rel_counts.items():
        print(f"  {t:15s}: {count:,}")
    print(f"\nTotal nodes        : {len(manifest['nodes']):,}")
    print(f"Total relationships: {len(manifest['relationships']):,}")
