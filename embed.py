"""
embed.py
========
Generate embeddings for all nodes that have a `text` field and no `embedding` yet.

WHY EMBEDDINGS:
  The Cursor-style retrieval engine works by:
    1. Embedding the user's query
    2. Vector searching Neo4j to find the closest nodes (entry points)
    3. Traversing the graph outward from those entry points
  Without embeddings, step 2 is impossible — there is nothing to search against.

WHICH NODES GET EMBEDDED:
  Shadow   → text = the raw chunk (1,000 tokens). Fine-grained retrieval.
  Section  → text = title + first paragraph. Mid-level retrieval.
  Document → text = LLM-generated summary. Coarse routing ("which filing?").
  Company  → no text field, skipped.

WHY OPENAI text-embedding-3-small:
  - Already have the API key
  - 1536 dimensions, strong performance on financial text
  - Cheap: ~$0.02 per million tokens
  - For 2,588 shadow nodes at ~500 tokens avg = ~1.3M tokens ≈ $0.026 total

HOW IT WORKS:
  1. Query Neo4j for all nodes with text/summary but no embedding
  2. Batch embed via OpenAI API (up to 100 texts per call)
  3. Write embeddings back to Neo4j in batches
  4. Dump all embeddings to manifest/embeddings.parquet as a local backup
     (so you can wipe and reload Neo4j without paying for re-embedding)

IDEMPOTENT:
  Only nodes with embedding=null are processed. Safe to re-run if it fails mid-way.

PARQUET BACKUP:
  After embedding, all vectors are written to manifest/embeddings.parquet.
  Each row: (id, label, embedding). Load back with:
    df = pd.read_parquet("manifest/embeddings.parquet")

Usage:
  python embed.py

Requirements:
  pip install openai neo4j python-dotenv pyyaml pandas pyarrow
"""

import os
import yaml
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT      = os.path.abspath(os.path.dirname(__file__))
_cfg       = yaml.safe_load(open(os.path.join(_ROOT, "config.yaml")))

EMB_MODEL  = _cfg["embedding"]["model"]
BATCH_SIZE = _cfg["embedding"]["batch_size"]

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")

# Filename encodes model + dimensions so switching models never overwrites old vectors.
# e.g. manifest/embeddings_text-embedding-3-small_1536d.parquet
_dims = _cfg["embedding"]["dimensions"]
PARQUET_PATH = os.path.join(
    _ROOT, "manifest",
    f"embeddings_{EMB_MODEL}_{_dims}d.parquet"
)

_openai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI embeddings API for a batch of texts.
    Returns a list of float vectors in the same order as input.

    WHY BATCHING:
      Each API call has network overhead. Sending 100 texts at once is
      ~100x more efficient than 100 separate calls.
    """
    response = _openai.embeddings.create(model=EMB_MODEL, input=texts)
    # API returns results sorted by index, so order is preserved
    return [r.embedding for r in sorted(response.data, key=lambda x: x.index)]


def batches(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ---------------------------------------------------------------------------
# Neo4j read/write
# ---------------------------------------------------------------------------

def fetch_unembedded(session, label: str) -> list[dict]:
    """
    Fetch all nodes of a given label that have no embedding yet.
    Returns list of {id, text} dicts where text is the field to embed.

    Shadow nodes embed their raw chunk text (field: text).
    Document/Section nodes embed their LLM summary (field: summary).

    WHY `embedding IS NULL`:
      Idempotency — if embed.py crashes halfway through, re-running it
      skips already-embedded nodes and picks up where it left off.
    """
    field = "text" if label == "Shadow" else "summary"
    result = session.run(f"""
        MATCH (n:{label})
        WHERE n.{field} IS NOT NULL AND n.embedding IS NULL
        RETURN n.id AS id, n.{field} AS text
    """)
    return [{"id": r["id"], "text": r["text"]} for r in result]


def write_embeddings(session, label: str, id_embedding_pairs: list[tuple]):
    """
    Write embeddings back to Neo4j for a batch of nodes.
    Uses UNWIND for efficiency — one round-trip per batch.
    """
    session.run(f"""
        UNWIND $rows AS row
        MATCH (n:{label} {{id: row.id}})
        SET n.embedding = row.embedding
    """, rows=[{"id": id_, "embedding": emb} for id_, emb in id_embedding_pairs])


# ---------------------------------------------------------------------------
# Parquet backup
# ---------------------------------------------------------------------------

def fetch_all_embeddings(session, label: str) -> list[dict]:
    """Fetch all embedded nodes of a label for the parquet backup."""
    result = session.run(f"""
        MATCH (n:{label})
        WHERE n.embedding IS NOT NULL
        RETURN n.id AS id, n.embedding AS embedding
    """)
    return [{"id": r["id"], "label": label, "embedding": r["embedding"]} for r in result]


def save_parquet(session):
    """
    Dump all embeddings from Neo4j to manifest/embeddings.parquet.

    WHY FETCH FROM NEO4J (not accumulate during embedding):
      embed_label skips already-embedded nodes — so if some nodes were embedded
      in a previous run, they'd be missing from an in-memory accumulator.
      Fetching from Neo4j captures everything regardless of which run produced it.
    """
    print("\nSaving embeddings to parquet backup...")
    rows = []
    for label in ("Document", "Section", "Shadow"):
        label_rows = fetch_all_embeddings(session, label)
        rows.extend(label_rows)
        print(f"  {label}: {len(label_rows)} rows")

    pd.DataFrame(rows).to_parquet(PARQUET_PATH, index=False)
    print(f"  Saved {len(rows)} embeddings → {PARQUET_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def embed_label(session, label: str):
    """Fetch, embed, and write back all unembedded nodes of a given label."""
    nodes = fetch_unembedded(session, label)
    if not nodes:
        print(f"  {label}: already embedded (or no text)")
        return

    print(f"  {label}: {len(nodes)} nodes to embed")
    done = 0

    for batch in batches(nodes, BATCH_SIZE):
        texts = [n["text"] for n in batch]
        vectors = embed_texts(texts)
        pairs = [(batch[i]["id"], vectors[i]) for i in range(len(batch))]
        write_embeddings(session, label, pairs)
        done += len(batch)
        print(f"    {done}/{len(nodes)}", end="\r")

    print(f"  {label}: {done} embedded    ")


def main():
    print(f"Embedding model: {EMB_MODEL}\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        # Embed in order: Document (coarse) → Section (mid) → Shadow (fine)
        embed_label(session, "Document")
        embed_label(session, "Section")
        embed_label(session, "Shadow")

        # Backup all embeddings to parquet so Neo4j can be wiped and reloaded
        # without paying for re-embedding.
        save_parquet(session)

    driver.close()
    print("\nDone. All nodes embedded.")
    print("Next: run cluster.py to assign cluster_id via Louvain.")


if __name__ == "__main__":
    main()
