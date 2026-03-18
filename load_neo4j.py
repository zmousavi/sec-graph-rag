"""
load_neo4j.py
=============
Load the manifest.json into Neo4j.

WHY THIS IS A SEPARATE SCRIPT FROM INGEST:
  ingest.py produces a manifest (plain JSON) that is pipeline-agnostic.
  This script is the only place that knows about Neo4j. That separation means
  you can regenerate the manifest without touching the database, or reload the
  database without re-running the pipeline.

WHAT IT DOES:
  1. Creates uniqueness constraints (safe to re-run — Neo4j ignores if they exist)
  2. Creates a vector index on Shadow nodes (for Cursor-style retrieval later)
  3. Batch-upserts all nodes using UNWIND + MERGE (idempotent — safe to re-run)
  4. Batch-upserts all relationships using UNWIND + MERGE

WHY MERGE NOT CREATE:
  MERGE = "create if not exists, match if it does." Running this script twice
  will not duplicate data. Safe for incremental loads when new filings are added.

WHY UNWIND (batch) NOT one-by-one:
  Sending 3,469 individual Cypher statements over the network is slow.
  UNWIND batches them into a single round-trip per node type. ~10x faster.

Usage:
  python load_neo4j.py

Requirements:
  pip install neo4j python-dotenv
"""

import json
import os
import yaml
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT       = os.path.abspath(os.path.dirname(__file__))
MANIFEST    = os.path.join(_ROOT, "manifest", "manifest.json")
NEO4J_URI   = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER  = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS  = os.getenv("NEO4J_PASSWORD", "")

_cfg        = yaml.safe_load(open(os.path.join(_ROOT, "config.yaml")))
DIMENSIONS  = _cfg["embedding"]["dimensions"]   # must match the embedding model

# How many nodes/rels to send per Cypher call.
# 500 is a safe default — large enough to be fast, small enough to avoid
# hitting Neo4j's transaction memory limit.
BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------

def create_schema(session):
    """
    Create uniqueness constraints and indexes.
    These are idempotent — Neo4j skips them silently if they already exist.

    WHY CONSTRAINTS:
      A uniqueness constraint on `id` ensures MERGE matches on the right node
      and also creates an implicit index, making lookups O(log n) not O(n).

    WHY VECTOR INDEX:
      The Cursor-style retrieval engine does vector search on Shadow nodes
      to find the entry points into the graph. The index must exist before
      embeddings are loaded (Step 5).
    """
    constraints = [
        "CREATE CONSTRAINT company_id  IF NOT EXISTS FOR (n:Company)  REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT section_id  IF NOT EXISTS FOR (n:Section)  REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT shadow_id   IF NOT EXISTS FOR (n:Shadow)   REQUIRE n.id IS UNIQUE",
    ]
    for q in constraints:
        session.run(q)
        print(f"  constraint: {q.split('FOR')[1].split('REQUIRE')[0].strip()}")

    # Vector indexes for Cursor-style semantic search.
    # All three node types with embeddings get indexes:
    #   Shadow   → fine retrieval (exact chunk)
    #   Section  → mid-level retrieval (which section?)
    #   Document → coarse routing (which document/filing?)
    # Dimensions are read from config.yaml so changing the embedding model
    # only requires updating config — not this file.
    for label, index_name in [("Shadow", "shadow_embedding"), ("Section", "section_embedding"), ("Document", "document_embedding")]:
        session.run(f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label}) ON n.embedding
            OPTIONS {{indexConfig: {{`vector.dimensions`: {DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}
        """)
        print(f"  vector index: {label}.embedding ({DIMENSIONS}d)")

    # Index on cluster_id for fast Louvain-based cluster routing.
    session.run("CREATE INDEX cluster_id IF NOT EXISTS FOR (n:Shadow) ON (n.cluster_id)")
    print("  index: Shadow.cluster_id")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def batches(lst, size):
    """Yield successive chunks of `lst` of length `size`."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def load_nodes(session, nodes, label, props):
    """
    Batch-upsert nodes of a given label.

    `props` = list of property keys to set from the manifest dict.
    Uses MERGE on `id` so re-runs are safe.
    """
    # Filter to only nodes of this type
    typed = [n for n in nodes if n["type"] == label]
    if not typed:
        return 0

    # Build SET clause dynamically from the requested properties
    set_clause = ", ".join(f"n.{p} = row.{p}" for p in props if p != "id")

    query = f"""
        UNWIND $rows AS row
        MERGE (n:{label} {{id: row.id}})
        SET {set_clause}
    """
    count = 0
    for batch in batches(typed, BATCH_SIZE):
        session.run(query, rows=batch)
        count += len(batch)
    return count


def load_relationships(session, rels, rel_type):
    """
    Batch-upsert relationships of a given type.
    MERGE prevents duplicates on re-run.
    """
    typed = [r for r in rels if r["type"] == rel_type]
    if not typed:
        return 0

    query = f"""
        UNWIND $rows AS row
        MATCH (a {{id: row.from}})
        MATCH (b {{id: row.to}})
        MERGE (a)-[:{rel_type}]->(b)
    """
    count = 0
    for batch in batches(typed, BATCH_SIZE):
        session.run(query, rows=batch)
        count += len(batch)
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading manifest: {MANIFEST}")
    with open(MANIFEST, "r") as f:
        manifest = json.load(f)

    nodes = manifest["nodes"]
    rels  = manifest["relationships"]
    print(f"  {len(nodes)} nodes, {len(rels)} relationships\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        # 1. Schema
        print("Setting up schema...")
        create_schema(session)
        print()

        # 2. Nodes (order matters: parents before children for cleaner MERGE)
        print("Loading nodes...")
        n = load_nodes(session, nodes, "Company",  ["id", "ticker", "updated_at"])
        print(f"  Company:  {n}")
        n = load_nodes(session, nodes, "Document", ["id", "ticker", "form_type", "period", "source_file", "summary", "updated_at"])
        print(f"  Document: {n}")
        n = load_nodes(session, nodes, "Section",  ["id", "doc_id", "title", "section_index", "summary", "updated_at"])
        print(f"  Section:  {n}")
        n = load_nodes(session, nodes, "Shadow",   ["id", "doc_id", "section_id", "chunk_index", "text", "token_count", "updated_at"])
        print(f"  Shadow:   {n}")
        print()

        # 3. Relationships
        print("Loading relationships...")
        r = load_relationships(session, rels, "HAS_DOCUMENT")
        print(f"  HAS_DOCUMENT: {r}")
        r = load_relationships(session, rels, "HAS_SECTION")
        print(f"  HAS_SECTION:  {r}")
        r = load_relationships(session, rels, "HAS_SHADOW")
        print(f"  HAS_SHADOW:   {r}")
        r = load_relationships(session, rels, "NEXT_CHUNK")
        print(f"  NEXT_CHUNK:   {r}")

    driver.close()
    print("\nDone. Graph loaded into Neo4j.")
    print("Next: run embed.py to generate embeddings for Shadow nodes.")


if __name__ == "__main__":
    main()
