"""
summarize.py
============
Generate summaries for Document and Section nodes that have no summary yet.

WHY THIS IS SEPARATE FROM INGEST:
  ingest.py is pure text processing — fast, offline, no API calls.
  Summarization requires OpenAI and adds latency (one LLM call per node).
  Keeping them separate means you can re-run ingest freely without API cost,
  and run summarize only when needed (e.g. after adding new filings).

WHAT IT DOES:
  1. Reads manifest.json
  2. For each Document with summary=null: reads source file from disk → LLM → writes summary
  3. For each Section with summary=null: reads full_text from manifest → LLM → writes summary
     (full_text is the complete pre-chunk section body written by ingest.py)
  4. Saves updated manifest.json

WHY SECTION SUMMARIES:
  Section text is used for mid-level embedding ("which section is this query about?").
  A proper LLM summary over the full section body gives much better embeddings than
  the title + first paragraph heuristic.

WHY full_text IN MANIFEST (not re-reading source file):
  ingest.py already split the document into sections. Storing full_text in the manifest
  means summarize.py gets the exact same section body ingest used — no re-splitting,
  no overlap artifacts from shadow chunks. full_text is manifest-only and never written
  to Neo4j (load_neo4j.py excludes it from the Section props list).

WHY MANIFEST ONLY (no Neo4j writes):
  summarize.py runs BEFORE load_neo4j.py. The manifest is the source of truth at this
  stage. load_neo4j.py picks up the completed manifest with summaries filled in.

IDEMPOTENT:
  Only processes nodes where summary is null. Safe to re-run.

Usage:
  python summarize.py

Requirements:
  pip install openai tiktoken python-dotenv pyyaml
"""

import os
import json
import yaml
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT      = os.path.abspath(os.path.dirname(__file__))
MANIFEST   = os.path.join(_ROOT, "manifest", "manifest.json")

_cfg       = yaml.safe_load(open(os.path.join(_ROOT, "config.yaml")))
MODEL      = _cfg["summarization"]["model"]
MAX_TOKENS = _cfg["summarization"]["input_tokens"]

_openai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_tokenizer = tiktoken.get_encoding("cl100k_base")

INPUT_DIRS = {
    "annual":    os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "annual"),
    "quarterly": os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "quarterly"),
    "def14a":    os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "def14a"),
    "8k":        os.path.join(_ROOT, "data", "processed", "sec_txt_clean", "8k"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_tokens: int) -> str:
    tokens = _tokenizer.encode(text)
    return text if len(tokens) <= max_tokens else _tokenizer.decode(tokens[:max_tokens])


def _llm(prompt: str, max_tokens: int) -> str:
    response = _openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def find_source_file(source_file: str) -> str | None:
    for dir_path in INPUT_DIRS.values():
        path = os.path.join(dir_path, source_file)
        if os.path.exists(path):
            return path
    return None

# ---------------------------------------------------------------------------
# Summary generators
# ---------------------------------------------------------------------------

def summarize_document(text: str, node: dict) -> str:
    """
    2-3 sentence summary of a full SEC filing.
    Reusable by the upload API — same function, same output format.
    """
    ticker    = node.get("ticker", "Unknown")
    form_type = node.get("form_type", "filing")
    period    = node.get("period", "unknown period")

    prompt = (
        f"You are summarizing a {form_type} SEC filing for {ticker} covering {period}. "
        f"Write a 2-3 sentence summary that captures: what the company does, "
        f"the key financial or operational highlights for this period, and any major risks "
        f"or strategic themes mentioned. Be factual and concise.\n\n"
        f"Document excerpt:\n{_truncate(text, MAX_TOKENS)}"
    )
    try:
        return _llm(prompt, max_tokens=150)
    except Exception as e:
        print(f"  [WARN] {e}")
        return f"{ticker} {form_type} filing for {period}."


def summarize_section(full_text: str, node: dict, doc: dict) -> str:
    """
    1-2 sentence summary of a single section (Item 1A, Item 7, etc.).
    full_text is the complete pre-chunk section body from the manifest.
    Reusable by the upload API — same function, same output format.
    """
    ticker    = doc.get("ticker", "Unknown")
    form_type = doc.get("form_type", "filing")
    period    = doc.get("period", "unknown period")
    title     = node.get("title", "section")

    prompt = (
        f"You are summarizing the '{title}' section of a {form_type} SEC filing "
        f"for {ticker} covering {period}. "
        f"Write 1-2 sentences capturing the key content of this section. "
        f"Be specific and factual — avoid generic phrases.\n\n"
        f"Section text:\n{_truncate(full_text, MAX_TOKENS)}"
    )
    try:
        return _llm(prompt, max_tokens=100)
    except Exception as e:
        print(f"  [WARN] {e}")
        return f"{title} from {ticker} {form_type} {period}."

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading manifest: {MANIFEST}")
    with open(MANIFEST) as f:
        manifest = json.load(f)

    nodes = manifest["nodes"]
    now   = datetime.now(timezone.utc).isoformat()

    # Build doc_id → Document node map (used by section summarization)
    doc_map = {n["id"]: n for n in nodes if n["type"] == "Document"}

    # --- Documents ---
    docs_pending = [n for n in nodes if n["type"] == "Document" and not n.get("summary")]
    if docs_pending:
        print(f"Documents to summarize: {len(docs_pending)}\n")
        for node in docs_pending:
            path = find_source_file(node.get("source_file", ""))
            if not path:
                print(f"  [SKIP] source file not found: {node.get('source_file')}")
                continue
            print(f"  {node['id']}...", end=" ", flush=True)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            summary = summarize_document(text, node)
            node["summary"]    = summary
            node["updated_at"] = now
            print("done")
    else:
        print("Documents: all summaries present.")

    # --- Sections ---
    secs_pending = [n for n in nodes if n["type"] == "Section" and not n.get("summary")]
    if secs_pending:
        print(f"\nSections to summarize: {len(secs_pending)}\n")
        for node in secs_pending:
            full_text = node.get("full_text", "")
            if not full_text:
                print(f"  [SKIP] no full_text on section: {node['id']}")
                continue
            doc = doc_map.get(node.get("doc_id", ""), {})
            print(f"  {node['id'][:70]}...", end=" ", flush=True)
            summary = summarize_section(full_text, node, doc)
            node["summary"]    = summary
            node["updated_at"] = now
            print("done")
    else:
        print("Sections: all summaries present.")

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest updated → {MANIFEST}")
    print("Next: run load_neo4j.py to load into Neo4j.")


if __name__ == "__main__":
    main()
