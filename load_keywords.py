"""
load_keywords.py
================
Write Keyword nodes and HAS_KEYWORD edges to Neo4j from manifest/keywords.json.

Pipeline position: run AFTER extract_keywords.py.

WHAT IT DOES:
  1. Creates a uniqueness constraint on Keyword.text (safe to re-run)
  2. Batch-MERGEs Keyword nodes (one per unique phrase)
  3. Batch-MERGEs HAS_KEYWORD edges: (Shadow)-[:HAS_KEYWORD {score}]->(Keyword)

WHY MERGE NOT CREATE:
  Idempotent — safe to re-run if you re-extract keywords and want to refresh.

Usage:
  python load_keywords.py
"""

import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

_ROOT        = os.path.abspath(os.path.dirname(__file__))
KEYWORDS_FILE = os.path.join(_ROOT, "manifest", "keywords.json")
BATCH_SIZE   = 500

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


def run(session, query, **params):
    session.run(query, **params)


def create_constraint(session):
    session.run(
        "CREATE CONSTRAINT keyword_text IF NOT EXISTS "
        "FOR (k:Keyword) REQUIRE k.text IS UNIQUE"
    )
    print("  Constraint: Keyword.text IS UNIQUE")


def load_keyword_nodes(session, unique_phrases: list[str]):
    """MERGE one Keyword node per unique phrase."""
    for i in range(0, len(unique_phrases), BATCH_SIZE):
        batch = [{"text": p} for p in unique_phrases[i:i + BATCH_SIZE]]
        session.run(
            "UNWIND $rows AS row MERGE (k:Keyword {text: row.text})",
            rows=batch,
        )
        print(f"  Keyword nodes: {min(i + BATCH_SIZE, len(unique_phrases))}/{len(unique_phrases)}")


def load_edges(session, edges: list[dict]):
    """MERGE HAS_KEYWORD edges: (Shadow)-[:HAS_KEYWORD {score}]->(Keyword)."""
    for i in range(0, len(edges), BATCH_SIZE):
        batch = edges[i:i + BATCH_SIZE]
        session.run(
            """
            UNWIND $rows AS row
            MATCH (sh:Shadow {id: row.shadow_id})
            MATCH (k:Keyword  {text: row.text})
            MERGE (sh)-[r:HAS_KEYWORD]->(k)
            SET r.score = row.score
            """,
            rows=batch,
        )
        print(f"  HAS_KEYWORD edges: {min(i + BATCH_SIZE, len(edges))}/{len(edges)}")


def main():
    print("=" * 60)
    print("Loading keywords into Neo4j")
    print("=" * 60)

    print(f"\n[1] Reading {KEYWORDS_FILE}...")
    with open(KEYWORDS_FILE, encoding="utf-8") as f:
        keywords: dict[str, list[dict]] = json.load(f)
    print(f"  {len(keywords)} chunks with keywords.")

    # Collect unique phrases and flat edge list
    unique_phrases: set[str] = set()
    edges: list[dict] = []
    for shadow_id, kws in keywords.items():
        for kw in kws:
            unique_phrases.add(kw["text"])
            edges.append({"shadow_id": shadow_id, "text": kw["text"], "score": kw["score"]})

    print(f"  {len(unique_phrases)} unique Keyword nodes to create.")
    print(f"  {len(edges)} HAS_KEYWORD edges to create.")

    with driver.session() as session:
        print("\n[2] Creating constraint...")
        create_constraint(session)

        print("\n[3] Merging Keyword nodes...")
        load_keyword_nodes(session, list(unique_phrases))

        print("\n[4] Merging HAS_KEYWORD edges...")
        load_edges(session, edges)

    driver.close()
    print("\nDone.")
    print("  Next: add traverse_keyword() to retrieve.py to use HAS_KEYWORD in RAG.")


if __name__ == "__main__":
    main()
