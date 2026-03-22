"""
cluster.py
==========
Offline preprocessing — run ONCE (or after data changes).

Assigns a cluster_id to every Section and Shadow node so the retrieval
engine can route queries to the right neighborhood instead of searching
the entire graph.

MUST RUN AFTER: load_neo4j.py, embed.py, extract_keywords.py, load_keywords.py
  Phase 1b uses HAS_KEYWORD edges already loaded in Neo4j. Run extract_keywords.py
  and load_keywords.py before cluster.py or Phase 1b will produce no edges.

WHY NOT LOUVAIN ON STRUCTURAL EDGES:
  The financial graph is a tree (Company → Document → Section → Shadow).
  Louvain on a tree produces one cluster per branch — each document becomes
  its own community, which is useless for cross-document routing.
  Instead we build a similarity graph from Section embeddings first, then
  run Louvain on those similarity edges to get meaningful topic clusters.

WHY SECTION LEVEL (NOT SHADOW):
  Shadow nodes are 1,000-token chunks — too granular. Chunks from the same
  section are nearly identical in embedding space and would cluster together
  trivially. Section embeddings (LLM summaries) are richer and produce
  meaningful clusters like "Risk Factors", "Revenue/MD&A", "Liquidity".
  cluster_id is then propagated from Section down to its Shadow children.

HOW IT WORKS:
  Phase 1 — Build embedding similarity edges
    Load Section embeddings from manifest/embeddings.parquet.
    Compute cosine similarity between all pairs of Section nodes.
    Write [:SIMILAR_TO {score}] edges for pairs above SIMILARITY_THRESHOLD.

  Phase 1b — Build keyword co-occurrence edges
    Query Neo4j for Section pairs whose Shadow children share specific keywords
    via HAS_KEYWORD edges (written by load_keywords.py).
    Write [:SIMILAR_TO] edges for pairs sharing >= KEYWORD_EDGE_MIN_SHARED keywords
    where no embedding-based edge already exists.
    WHY THIS MATTERS: embedding similarity captures semantic closeness but misses
    company-specific terminology. "Hindenburg" only appears in NKLA chunks —
    keyword co-occurrence bonds NKLA allegation sections together so Louvain keeps
    them in a tight NKLA-specific cluster rather than merging them into a generic
    EV legal cluster with TSLA. Specific named entities (people, organisations,
    products) that appear in only one company's filings create strong intra-company
    bonds that embeddings alone cannot produce.

  Phase 2 — Run Louvain (GDS)
    Project Section nodes + all SIMILAR_TO edges (both sources) into GDS memory.
    Run gds.louvain.write → writes cluster_id to every Section node.

  Phase 3 — Propagate cluster_id down to Shadow nodes
    Each Shadow inherits cluster_id from its parent Section via HAS_SHADOW.

  Phase 4 — Verify and print distribution
    Print how many nodes landed in each cluster.
    Flag if coverage is incomplete.

TUNING THE THRESHOLD:
  SIMILARITY_THRESHOLD = 0.78 (current).
  Too low  (< 0.75) → one giant cluster, no routing signal.
  Too high (> 0.92) → most sections are singletons, also useless.
  If the distribution shows one cluster with >50% of nodes, lower it.
  If most clusters have 1-2 nodes, raise it.
  Threshold history:
    0.82 → 188 clusters, 66 singletons (too many isolated sections)
    0.78 → current (121 clusters, modularity=0.757)

KEYWORD_EDGE_MIN_SHARED = 8 (current).
  Sections must share at least 8 keyword nodes to get a co-occurrence edge.
  Too low  (1-3) → generic bankruptcy/distress vocabulary ("going concern",
    "restructuring") creates cross-company noise edges that merge FSR + NKLA
    into the same cluster, making disambiguation worse not better.
  Too high (15+) → only near-identical sections connect, no new signal.
  Threshold history:
    3 → FSR+NKLA merged into cluster 2519 (287 sections) — too low
    8 → current

Requirements:
  pip install neo4j python-dotenv scikit-learn pandas pyarrow numpy

Usage:
  python cluster.py
"""

import os
import yaml
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.dirname(__file__))
_cfg  = yaml.safe_load(open(os.path.join(_ROOT, "config.yaml")))

# Match the filename produced by embed.py so the two scripts always agree.
_emb_model = _cfg["embedding"]["model"]
_emb_dims  = _cfg["embedding"]["dimensions"]
PARQUET_PATH = os.path.join(
    _ROOT, "manifest",
    f"embeddings_{_emb_model}_{_emb_dims}d.parquet"
)
# Filename encodes threshold + section count so different corpus sizes and
# threshold values never overwrite each other.
# e.g.  manifest/similarity_edges_t0.78_n1069.parquet
def _edges_path(threshold: float, n_sections: int) -> str:
    return os.path.join(_ROOT, "manifest", f"similarity_edges_t{threshold}_n{n_sections}.parquet")

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")

# Cosine similarity threshold for SIMILAR_TO edges.
# Pairs of Section nodes above this score get connected.
#
# Threshold history:
#   0.82 → 188 clusters, 66 singletons (too many isolated sections)
#   0.78 → current (121 clusters, modularity=0.757)
SIMILARITY_THRESHOLD = 0.78

# Minimum shared keywords between two sections to create a co-occurrence edge.
# Sections must share at least this many Keyword nodes (via their Shadow children)
# to receive a SIMILAR_TO edge from Phase 1b.
KEYWORD_EDGE_MIN_SHARED = 8

# GDS graph projection name (in-memory, dropped after use).
GRAPH_NAME = "financial-similarity-graph"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))


# ===========================================================================
# Phase 1 — Build similarity edges
# ===========================================================================

def load_section_embeddings() -> tuple[list, np.ndarray]:
    """Load Section node IDs and embedding vectors from parquet."""
    df = pd.read_parquet(PARQUET_PATH)
    sections = df[df["label"] == "Section"].copy()
    print(f"  Loaded {len(sections)} Section embeddings from parquet.")

    ids = sections["id"].tolist()
    # Each embedding is stored as a list; stack into a matrix.
    matrix = np.vstack(sections["embedding"].values)
    return ids, matrix


def save_edges(edges: list, threshold: float, n_sections: int):
    """Persist edges to parquet for future re-runs."""
    path = _edges_path(threshold, n_sections)
    df = pd.DataFrame(edges, columns=["id_a", "id_b", "score"])
    df.to_parquet(path, index=False)
    print(f"  Saved {len(edges)} edges → {os.path.basename(path)}")


def load_edges(threshold: float, n_sections: int) -> list | None:
    """Return cached edges if the matching parquet exists, else None."""
    path = _edges_path(threshold, n_sections)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    edges = list(df.itertuples(index=False, name=None))
    print(f"  Loaded {len(edges)} edges from {os.path.basename(path)} (skipping recompute).")
    return edges


def build_similarity_edges(session, ids: list, matrix: np.ndarray):
    """
    Compute pairwise cosine similarity and write SIMILAR_TO edges to Neo4j
    for all pairs above SIMILARITY_THRESHOLD.

    Saves edges to manifest/similarity_edges_t{threshold}_n{sections}.parquet
    so future re-runs skip the O(N²) cosine computation.

    Uses batched writes to avoid a single massive transaction.
    """
    n = len(ids)

    # Try loading from cache first.
    edges = load_edges(SIMILARITY_THRESHOLD, n)

    if edges is None:
        print(f"  Computing cosine similarity for {n} sections...")
        sim_matrix = cosine_similarity(matrix)  # shape: (N, N)

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= SIMILARITY_THRESHOLD:
                    edges.append((ids[i], ids[j], round(score, 4)))

        print(f"  Found {len(edges)} pairs above threshold {SIMILARITY_THRESHOLD}.")
        save_edges(edges, SIMILARITY_THRESHOLD, n)
    else:
        print(f"  Found {len(edges)} pairs above threshold {SIMILARITY_THRESHOLD}.")

    if not edges:
        print("  WARNING: No edges found. Lower SIMILARITY_THRESHOLD and re-run.")
        return 0

    # Delete any old SIMILAR_TO edges before writing fresh ones.
    session.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")
    print("  Cleared old SIMILAR_TO edges.")

    # Write in batches of 500.
    BATCH = 500
    written = 0
    for start in range(0, len(edges), BATCH):
        batch = edges[start : start + BATCH]
        params = [{"a": a, "b": b, "score": s} for a, b, s in batch]
        session.run("""
            UNWIND $rows AS row
            MATCH (a:Section {id: row.a})
            MATCH (b:Section {id: row.b})
            MERGE (a)-[:SIMILAR_TO {score: row.score}]->(b)
        """, rows=params)
        written += len(batch)

    print(f"  Wrote {written} SIMILAR_TO edges to Neo4j.")
    return written


# ===========================================================================
# Phase 1b — Build keyword co-occurrence edges
# ===========================================================================

def build_keyword_cooccurrence_edges(session) -> int:
    """
    Add SIMILAR_TO edges between Section pairs whose Shadow children share
    specific keywords via HAS_KEYWORD edges.

    Only creates edges where none already exist from Phase 1 embedding similarity —
    fills in topical bonds that cosine similarity misses. The key benefit is
    company-specific terminology: a keyword like "Hindenburg" only appears in NKLA
    chunks, so keyword co-occurrence creates strong intra-NKLA bonds that keep
    NKLA allegation sections in a tight cluster instead of merging them with
    generic EV legal content from TSLA or GM.

    Score = min(shared_keyword_count / 10.0, 1.0), normalized to [0, 1].

    Requires HAS_KEYWORD edges to be loaded (run extract_keywords.py +
    load_keywords.py before cluster.py).
    """
    print("  Querying keyword co-occurrence between sections...")
    result = session.run("""
        MATCH (a:Section)-[:HAS_SHADOW]->(sa:Shadow)-[:HAS_KEYWORD]->(k:Keyword)
              <-[:HAS_KEYWORD]-(sb:Shadow)<-[:HAS_SHADOW]-(b:Section)
        WHERE a.id < b.id
        WITH a, b, count(DISTINCT k) AS shared_count
        WHERE shared_count >= $min_shared
          AND NOT (a)-[:SIMILAR_TO]-(b)
        RETURN a.id AS id_a, b.id AS id_b, shared_count
    """, min_shared=KEYWORD_EDGE_MIN_SHARED)

    pairs = result.data()
    print(f"  Found {len(pairs)} keyword co-occurrence pairs (no existing SIMILAR_TO edge).")

    if not pairs:
        print("  Skipping — no pairs found. Ensure extract_keywords.py + load_keywords.py have been run.")
        return 0

    BATCH = 500
    written = 0
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start : start + BATCH]
        params = [{
            "a": r["id_a"],
            "b": r["id_b"],
            "score": round(min(r["shared_count"] / 10.0, 1.0), 4)
        } for r in batch]
        session.run("""
            UNWIND $rows AS row
            MATCH (a:Section {id: row.a})
            MATCH (b:Section {id: row.b})
            MERGE (a)-[:SIMILAR_TO {score: row.score}]->(b)
        """, rows=params)
        written += len(batch)

    print(f"  Wrote {written} keyword co-occurrence SIMILAR_TO edges.")
    return written


# ===========================================================================
# Phase 2 — Run Louvain via GDS
# ===========================================================================

def drop_graph_if_exists(session):
    result = session.run(
        "CALL gds.graph.exists($name) YIELD exists", name=GRAPH_NAME
    )
    if result.single()["exists"]:
        session.run("CALL gds.graph.drop($name)", name=GRAPH_NAME)
        print(f"  Dropped existing GDS projection '{GRAPH_NAME}'.")


def project_graph(session):
    """Project Section nodes + SIMILAR_TO edges into GDS memory."""
    print("  Projecting graph into GDS memory...")
    result = session.run("""
        CALL gds.graph.project(
            $name,
            'Section',
            {
                SIMILAR_TO: { orientation: 'UNDIRECTED' }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
    """, name=GRAPH_NAME)
    row = result.single()
    print(f"  Projected '{row['graphName']}': "
          f"{row['nodeCount']} nodes, {row['relationshipCount']} relationships.")


def run_louvain(session):
    """Run Louvain and write cluster_id to every Section node."""
    print("  Running Louvain community detection...")
    result = session.run("""
        CALL gds.louvain.write($name, {
            writeProperty: 'cluster_id',
            nodeLabels: ['Section']
        })
        YIELD communityCount, modularity, ranLevels
    """, name=GRAPH_NAME)
    row = result.single()
    print(f"  Communities found: {row['communityCount']}")
    print(f"  Modularity score:  {row['modularity']:.4f}  (higher = better separation)")
    print(f"  Louvain levels:    {row['ranLevels']}")
    return row["communityCount"]


# ===========================================================================
# Phase 3 — Propagate cluster_id from Section → Shadow
# ===========================================================================

def propagate_to_shadows(session):
    """
    Copy cluster_id from each Section to its child Shadow nodes.
    Shadow nodes do not appear in the similarity graph but need cluster_id
    so the retrieval engine can filter vector search by cluster.
    """
    print("  Propagating cluster_id from Section → Shadow...")
    result = session.run("""
        MATCH (sec:Section)-[:HAS_SHADOW]->(sh:Shadow)
        WHERE sec.cluster_id IS NOT NULL
        SET sh.cluster_id = sec.cluster_id
        RETURN count(sh) AS updated
    """)
    updated = result.single()["updated"]
    print(f"  Set cluster_id on {updated} Shadow nodes.")
    return updated


# ===========================================================================
# Phase 4 — Verify and print distribution
# ===========================================================================

def print_cluster_distribution(session):
    print("\n  Cluster distribution (sections per cluster):")
    result = session.run("""
        MATCH (s:Section)
        WHERE s.cluster_id IS NOT NULL
        RETURN s.cluster_id AS cluster_id, count(*) AS section_count
        ORDER BY section_count DESC
        LIMIT 30
    """)
    rows = result.data()
    total = sum(r["section_count"] for r in rows)
    for row in rows:
        pct = row["section_count"] / total * 100 if total else 0
        bar = "#" * min(row["section_count"], 50)
        print(f"    Cluster {row['cluster_id']:>5}:  "
              f"{row['section_count']:>4} sections  ({pct:.1f}%)  {bar}")

    # Warn if one cluster dominates.
    if rows and rows[0]["section_count"] / total > 0.5:
        print("\n  WARNING: Top cluster holds >50% of sections.")
        print("  Consider lowering SIMILARITY_THRESHOLD and re-running.")


def verify_coverage(session):
    result = session.run("""
        MATCH (s:Section)
        RETURN
            count(s)              AS total,
            count(s.cluster_id)   AS with_cluster,
            count(s) - count(s.cluster_id) AS missing
    """)
    row = result.single()
    print(f"\n  Section coverage: {row['with_cluster']}/{row['total']} "
          f"({row['missing']} missing)")

    result = session.run("""
        MATCH (s:Shadow)
        RETURN
            count(s)              AS total,
            count(s.cluster_id)   AS with_cluster,
            count(s) - count(s.cluster_id) AS missing
    """)
    row = result.single()
    print(f"  Shadow  coverage: {row['with_cluster']}/{row['total']} "
          f"({row['missing']} missing)")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=== Louvain Community Detection ===\n")

    # Phase 1
    print("Phase 1 — Building similarity edges")
    ids, matrix = load_section_embeddings()
    with driver.session() as session:
        edge_count = build_similarity_edges(session, ids, matrix)

    if edge_count == 0:
        print("\nAborting — no edges to cluster on. Adjust SIMILARITY_THRESHOLD.")
        driver.close()
        exit(1)

    # Phase 1b — disabled: keyword co-occurrence edges hurt cluster quality for
    # distressed EV companies (FSR + NKLA share generic bankruptcy vocabulary,
    # causing them to merge into the same cluster regardless of threshold).
    # HAS_KEYWORD edges are used only at retrieval time (traverse_keyword in retrieve.py).
    # print("\nPhase 1b — Adding keyword co-occurrence edges")
    # with driver.session() as session:
    #     build_keyword_cooccurrence_edges(session)

    # Phase 2
    print("\nPhase 2 — Running Louvain (GDS)")
    with driver.session() as session:
        drop_graph_if_exists(session)
        project_graph(session)
        community_count = run_louvain(session)
        session.run("CALL gds.graph.drop($name)", name=GRAPH_NAME)
        print(f"  Dropped GDS projection '{GRAPH_NAME}'.")

    # Phase 3
    print("\nPhase 3 — Propagating cluster_id to Shadow nodes")
    with driver.session() as session:
        propagate_to_shadows(session)

    # Phase 4
    print("\nPhase 4 — Verification")
    with driver.session() as session:
        print_cluster_distribution(session)
        verify_coverage(session)

    driver.close()
    print("\nDone. Re-run after adding new documents or changing SIMILARITY_THRESHOLD.")
