"""
retrieve.py
===========
Retrieval Orchestrator — Step 8 of the pipeline.

Two modes run on every question for side-by-side comparison:

  RAG (baseline)
    Embed question → metadata filter → global shadow vector search
    → concatenate raw chunk text → LLM answer.
    This is what a standard vector-DB RAG system would do.

  Graph RAG
    Embed question → metadata filter → global vector search → cluster routing
    → cluster-scoped vector search → graph traversal up (Section + Document +
    Company) !!!!!!!!!!!!!!!!!!!!!!!!!!! → structured context per result → LLM answer + supporting paths.

WHAT MAKES THE COMPARISON FAIR:
  Both modes use identical:
    - Embedding model
    - Metadata detection (company / year / form_type extracted from question)
    - Metadata filters applied to shadow vector search
    - Top-K shadows retrieved
  The ONLY difference is what the LLM receives:
    RAG      → raw chunk text only
    Graph RAG → chunk text + section title + section summary +
                ticker + form_type + period (from graph traversal)
  This isolates the value of graph context over raw chunks.

METADATA DETECTION:
  Simple keyword matching against known tickers and company names.
  No NLP dependency. Covers all 5 companies in config.yaml.
  Detected tickers and years are applied as WHERE filters on vector search
  so both modes search the same restricted subgraph.

OUTPUT CONTRACT:
  Both modes return a RetrievalResult with:
    answer           — LLM-generated string
    supporting_paths — list of {node_ids, edge_types, score} (empty for RAG)
    clusters_used    — list of cluster_ids (None for RAG)
    tickers_detected — list of tickers parsed from question
    filters_applied  — dict of active WHERE filters
    cache_status     — "miss" (Redis added in Step 9)
    mode             — "rag" | "graph_rag"
    latency_breakdown — per-step timing in ms

RESULTS:
  Each run saves results/results_{timestamp}.json with both modes
  side by side for every question.

Requirements:
  pip install openai neo4j python-dotenv pyyaml

Usage:
  python retrieve.py
"""

import os
import re
import json
import time
import random
import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

try:
    from sentence_transformers import CrossEncoder as _CrossEncoderClass
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False

# Lazy-loaded; first call triggers model download (~80 MB, cached after that).
_cross_encoder = None

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.dirname(__file__))
_cfg  = yaml.safe_load(open(os.path.join(_ROOT, "config.yaml")))

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")

EMB_MODEL  = _cfg["embedding"]["model"]       # text-embedding-3-small
EMB_DIMS   = _cfg["embedding"]["dimensions"]  # 1536
LLM_MODEL  = _cfg["summarization"]["model"]   # gpt-4o-mini

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many shadows to retrieve in each search pass.
VECTOR_TOP_K_GLOBAL  = 20   # global pass — used for cluster routing
CHUNKS_PER_CLUSTER   = 6    # top chunks kept per cluster (6 × 6 clusters = 36 anchors)
VECTOR_TOP_K_RAG     = 18   # flat retrieval for RAG baseline — matches Graph RAG anchor count

# How many top clusters to route to after global search.
CLUSTER_TOP_N = 6

# Minimum results before falling back to global search.
MIN_RESULTS_FALLBACK = 3

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Known entities for metadata detection
# Maps every recognizable name/ticker → canonical ticker stored in Neo4j
# ---------------------------------------------------------------------------

TICKER_MAP = {
    "tesla":   "TSLA", "tsla":  "TSLA",
    "ford":    "F",    "f":     "F",
    "gm":      "GM",   "general motors": "GM",
    "rivian":  "RIVN", "rivn":  "RIVN",
    "lucid":   "LCID", "lcid":  "LCID",
}

FORM_MAP = {
    "annual": "10-K", "10-k": "10-K", "10k": "10-K",
    "quarterly": "10-Q", "10-q": "10-Q", "10q": "10-Q",
}


# ===========================================================================
# Output contract
# ===========================================================================

@dataclass
class RetrievalResult:
    answer:           str
    supporting_paths: list
    clusters_used:    list
    tickers_detected: list
    filters_applied:  dict
    cache_status:     str
    mode:             str
    latency_breakdown: dict = field(default_factory=dict)

    def print_summary(self):
        tag = "*** CACHE HIT ***" if self.cache_status == "hit" else "cache miss"
        print(f"  mode:     {self.mode}")
        print(f"  cache:    {tag}")
        print(f"  tickers:  {self.tickers_detected or 'none (all companies)'}")
        print(f"  filters:  {self.filters_applied or 'none'}")
        print(f"  clusters: {self.clusters_used}")
        print(f"  paths:    {len(self.supporting_paths)}")
        lat = self.latency_breakdown
        print(f"  latency:  embed={lat.get('embed_ms',0):.0f}ms  "
              f"search={lat.get('search_ms',0):.0f}ms  "
              f"traverse={lat.get('traverse_ms',0):.0f}ms  "
              f"rerank={lat.get('rerank_ms',0):.0f}ms  "
              f"llm={lat.get('llm_ms',0):.0f}ms  "
              f"total={lat.get('total_ms',0):.0f}ms")

    def to_dict(self, question: str) -> dict:
        return {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "mode":              self.mode,
            "question":          question,
            "answer":            self.answer,
            "cache_status":      self.cache_status,
            "tickers_detected":  self.tickers_detected,
            "filters_applied":   self.filters_applied,
            "clusters_used":     self.clusters_used,
            "supporting_paths":  self.supporting_paths,
            "latency_breakdown": self.latency_breakdown,
        }


# ===========================================================================
# Step 1 — Metadata detection
# ===========================================================================

def detect_filters(question: str) -> dict:
    """
    Extract tickers, years, and form_type from a question using keyword matching.

    Returns a filters dict consumed by vector search Cypher queries:
      {
        "tickers":   ["TSLA", "F"],   # empty = no company filter
        "years":     ["2023"],         # empty = no year filter
        "form_type": "10-K",           # None = no form_type filter
      }

    No NLP — pure string matching. Fast and deterministic.
    Multi-word names ("general motors") are checked before single tokens
    so they are not split into partial matches.
    Single-character tickers (e.g. "F") use word-boundary matching so they
    don't match every word containing that letter ("factors", "for", etc.).
    """
    q = question.lower()

    # Multi-word names first (order matters — check before single tokens).
    tickers = []
    for name, ticker in TICKER_MAP.items():
        if len(name) == 1:
            # Single-char keys need word-boundary match to avoid false positives.
            if re.search(rf"\b{re.escape(name)}\b", q) and ticker not in tickers:
                tickers.append(ticker)
        else:
            if name in q and ticker not in tickers:
                tickers.append(ticker)

    # Years: match 4-digit years in range 2020-2026.
    years = re.findall(r"\b(202[0-6])\b", question)

    # Form type.
    form_type = None
    for keyword, ft in FORM_MAP.items():
        if keyword in q:
            form_type = ft
            break

    return {
        "tickers":   tickers,
        "years":     years,
        "form_type": form_type,
    }


# ===========================================================================
# Step 2 — Embedding
# ===========================================================================

def embed(text: str) -> list[float]:
    response = client.embeddings.create(model=EMB_MODEL, input=[text], dimensions=EMB_DIMS)
    return response.data[0].embedding


# ===========================================================================
# Step 3 — Shadow vector search (shared by RAG and Graph RAG)
# ===========================================================================

def search_shadows(tx, q_vec: list, filters: dict, top_k: int,
                   include_text: bool = True) -> list:
    """
    Vector search on Shadow nodes with optional metadata filters.

    include_text=True  — returns full columns for RAG/traversal (text, section_title, ticker, etc.)
    include_text=False — returns only shadow_id, cluster_id, score for cheap cluster routing pass.
    """
    where_clauses = []
    params = {"q_vec": q_vec, "top_k": top_k}

    if filters.get("tickers"):
        where_clauses.append("doc.ticker IN $tickers")
        params["tickers"] = filters["tickers"]
    if filters.get("years"):
        year_conditions = " OR ".join(
            [f"doc.period CONTAINS $year{i}" for i in range(len(filters["years"]))]
        )
        where_clauses.append(f"({year_conditions})")
        for i, y in enumerate(filters["years"]):
            params[f"year{i}"] = y
    if filters.get("form_type"):
        where_clauses.append("doc.form_type = $form_type")
        params["form_type"] = filters["form_type"]

    where_str = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    if include_text:
        return_clause = """
               sh.id          AS shadow_id,
               sh.text        AS text,
               sh.cluster_id  AS cluster_id,
               score,
               sec.title      AS section_title,
               doc.ticker     AS ticker,
               doc.form_type  AS form_type,
               doc.period     AS period"""
    else:
        return_clause = """
               sh.id          AS shadow_id,
               sh.cluster_id  AS cluster_id,
               score"""

    result = tx.run(f"""
        CALL db.index.vector.queryNodes('shadow_embedding', $top_k * 3, $q_vec)
        YIELD node AS sh, score
        MATCH (sec:Section)-[:HAS_SHADOW]->(sh)
        MATCH (doc:Document)-[:HAS_SECTION]->(sec)
        {where_str}
        RETURN {return_clause}
        ORDER BY score DESC
        LIMIT $top_k
    """, **params)
    return result.data()


def get_top_clusters(seeds: list, top_n: int) -> list:
    """
    Given seed shadows from global search, return the top_n cluster_ids
    ranked by MEAN similarity score within each cluster.

    CHANGED from sum → mean (2026-03-18):
      Sum favoured high-volume companies (e.g. NKLA with 2000+ chunks) because
      more seeds from that company accumulate a higher total even when individual
      scores are mediocre. Mean ranks clusters by how relevant their chunks are
      on average, regardless of cluster size. Revert to sum if mean turns out to
      under-select large clusters that genuinely dominate the answer.

    Clusters with null cluster_id (singletons) are excluded.
    """
    cluster_scores: dict = {}
    cluster_counts: dict = {}
    for row in seeds:
        cid = row.get("cluster_id")
        if cid is None:
            continue
        cluster_scores[cid] = cluster_scores.get(cid, 0.0) + row["score"]
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
    cluster_means = {cid: cluster_scores[cid] / cluster_counts[cid]
                     for cid in cluster_scores}
    ranked = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:top_n]]


# ===========================================================================
# Step 4G — Graph RAG: true cluster-scoped search via embedding fetch + numpy
# ===========================================================================

CLUSTER_FETCH_CAP = 5000  # max shadows fetched per cluster before numpy scoring


def fetch_shadows_in_clusters(tx, cluster_ids: list, filters: dict,
                               seed_ids: set = None) -> list:
    """
    Fetch ALL shadows (with embeddings) from the selected clusters.
    Metadata filters applied here to stay consistent with global search.
    Embeddings are returned so Python can score them against q_vec.

    When the result set exceeds CLUSTER_FETCH_CAP, seed-biased sampling is used:
    shadows that were already identified as relevant by the global vector search
    (seed_ids) are always kept; remaining slots are filled randomly from the rest.
    This prevents the key chunks that scored well in the index from being evicted
    by pure random sampling — the original bug that caused Q14 to miss the NKLA
    Hindenburg chunk (cluster 2826 had 1819 shadows; only 27.6% chance of survival
    with the old random.sample(rows, 1000) approach).
    """
    where_clauses = ["sh.cluster_id IN $cluster_ids"]
    params = {"cluster_ids": cluster_ids}

    if filters.get("tickers"):
        where_clauses.append("doc.ticker IN $tickers")
        params["tickers"] = filters["tickers"]
    if filters.get("years"):
        year_conditions = " OR ".join(
            [f"doc.period CONTAINS $year{i}" for i in range(len(filters["years"]))]
        )
        where_clauses.append(f"({year_conditions})")
        for i, y in enumerate(filters["years"]):
            params[f"year{i}"] = y
    if filters.get("form_type"):
        where_clauses.append("doc.form_type = $form_type")
        params["form_type"] = filters["form_type"]

    where_str = "WHERE " + " AND ".join(where_clauses)

    result = tx.run(f"""
        MATCH (sh:Shadow)
        MATCH (sec:Section)-[:HAS_SHADOW]->(sh)
        MATCH (doc:Document)-[:HAS_SECTION]->(sec)
        {where_str}
        RETURN sh.id          AS shadow_id,
               sh.text        AS text,
               sh.embedding   AS embedding,
               sh.cluster_id  AS cluster_id,
               sec.title      AS section_title,
               doc.ticker     AS ticker,
               doc.form_type  AS form_type,
               doc.period     AS period
    """, **params)
    rows = result.data()
    if len(rows) > CLUSTER_FETCH_CAP:
        if seed_ids:
            priority = [r for r in rows if r["shadow_id"] in seed_ids]
            rest     = [r for r in rows if r["shadow_id"] not in seed_ids]
            n_fill   = max(0, CLUSTER_FETCH_CAP - len(priority))
            rows = priority + random.sample(rest, min(n_fill, len(rest)))
        else:
            rows = random.sample(rows, CLUSTER_FETCH_CAP)
    return rows


def score_and_rank(candidates: list, q_vec: list, top_k: int) -> list:
    """
    Compute cosine similarity between q_vec and each candidate's embedding.
    Returns top_k candidates sorted by score descending, with score injected.

    SCORE FORMULA — (1 + cosine) / 2:
      Neo4j's vector index returns scores in this form (maps cosine [-1,1] → [0,1]).
      We apply the same formula here so numpy scores are directly comparable to
      the index scores shown in supporting_paths and used for cluster routing.
      Ranking is unaffected (the transformation is monotonic), but the numbers
      now read consistently: e.g. 0.763 from the index ↔ 0.763 from numpy.
      Verified 2026-03-18: numpy raw cosine 0.527 → (1+0.527)/2 = 0.763 ✓
    """
    if not candidates:
        return []
    q = np.array(q_vec, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    scored = []
    for row in candidates:
        emb = row.get("embedding")
        if emb is None:
            continue
        v = np.array(emb, dtype=np.float32)
        v_norm = v / (np.linalg.norm(v) + 1e-9)
        score = (1.0 + float(np.dot(q_norm, v_norm))) / 2.0
        scored.append({**row, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank(question: str, rows: list, text_key: str) -> list:
    """
    Cross-encoder reranking (Step 5 of the Cursor-Style spec).

    WHY THIS RUNS AFTER COSINE, NOT INSTEAD OF IT:
      A cross-encoder scores each (question, chunk) pair jointly via a full
      neural network forward pass — it reads both together, which makes it far
      more accurate than cosine similarity on near-tie cases (e.g. FSR vs NKLA
      chunks that are both about creditor negotiations).
      But it costs ~5-20 ms per pair, so running it on the full corpus (~7k+
      shadows) would take several minutes. Cosine via HNSW is O(log N) and runs
      in milliseconds — it narrows the field to ~20-30 candidates, which the
      cross-encoder re-scores in ~50-200 ms total.

    HOW THIS FIXES Q7/Q8/Q9:
      Cosine sees "creditor negotiations / debt acceleration / bankruptcy" and
      scores NKLA and FSR chunks nearly identically (near-tied embeddings). The
      cross-encoder sees the *full question* alongside each chunk and recognises
      that mid-2024 timing, the specific trigger (failed negotiations → default →
      acceleration), and the causal chain match FSR's 8-K — not NKLA's story.

    MODEL: ms-marco-MiniLM-L-6-v2
      Trained on MS MARCO passage relevance pairs. Fast (~1ms/pair on CPU).
      Returns a raw logit (higher = more relevant); not bounded to [0,1] but
      monotonically comparable within a query.

    Returns rows reordered by cross-encoder score descending.
    The 'score' field on each row is replaced with the cross-encoder score
    so it propagates correctly into supporting_paths and LLM context ordering.
    Rows where text_key is missing or empty are pushed to the end.
    """
    global _cross_encoder
    if not _CROSS_ENCODER_AVAILABLE or not rows:
        return rows

    if _cross_encoder is None:
        _cross_encoder = _CrossEncoderClass(CROSS_ENCODER_MODEL)

    valid   = [r for r in rows if r.get(text_key)]
    invalid = [r for r in rows if not r.get(text_key)]

    pairs = [(question, r[text_key]) for r in valid]
    ce_scores = _cross_encoder.predict(pairs)  # numpy array, float32

    for row, ce_score in zip(valid, ce_scores):
        row["score"] = float(ce_score)

    valid.sort(key=lambda r: r["score"], reverse=True)
    return valid + invalid


# ===========================================================================
# Step 5G — Graph RAG: traverse up from anchor shadows
# ===========================================================================

def traverse_up(tx, shadow_ids: list) -> list:
    """
    For each anchor shadow, traverse up to Section → Document → Company.
    Also fetches 1 NEXT_CHUNK neighbor per anchor for continuity context.

    Handles two document origins:
      Pipeline docs:  Company -[:HAS_DOCUMENT]-> Document -[:HAS_SECTION]-> Section
      Uploaded docs:  any node -[:LINKED_TO]-> Document -[:HAS_SECTION]-> Section
        (also have a [:HAS_DOCUMENT] edge written at upload time if ticker was provided)

    The Company lookup is OPTIONAL — uploaded docs with no ticker have no Company
    ancestor. In that case company_id is null and ticker falls back to doc.ticker.
    Without this, uploaded documents are silently dropped from Graph RAG results.
    """
    result = tx.run("""
        UNWIND $shadow_ids AS sid
        MATCH (sh:Shadow {id: sid})
        MATCH (sec:Section)-[:HAS_SHADOW]->(sh)
        MATCH (doc:Document)-[:HAS_SECTION]->(sec)
        OPTIONAL MATCH (co:Company)-[:HAS_DOCUMENT]->(doc)
        OPTIONAL MATCH (parent)-[:LINKED_TO]->(doc)
        RETURN sh.id            AS shadow_id,
               sh.text          AS shadow_text,
               sh.cluster_id    AS cluster_id,
               null             AS next_chunk_text,
               sec.id           AS section_id,
               sec.title        AS section_title,
               sec.summary      AS section_summary,
               doc.id           AS doc_id,
               doc.ticker       AS ticker,
               doc.form_type    AS form_type,
               doc.period       AS period,
               co.id            AS company_id,
               parent.id        AS contains_parent_id,
               labels(parent)[0] AS contains_parent_label
    """, shadow_ids=shadow_ids)
    return result.data()


# ===========================================================================
# Step 5G-b — Follow SIMILAR_TO edges to neighboring sections (actual graph hop)
# ===========================================================================

def traverse_similar(tx, section_ids: list, already_seen: set, top_n_per_anchor: int = 3) -> list:
    """
    Follow SIMILAR_TO edges from each anchor section to its neighbors.

    SIMILAR_TO edges are built offline in cluster.py: pairwise cosine on all Section
    embeddings, threshold=0.78 → ~7 neighbors per section on average. This is the
    primary graph traversal for pipeline docs (10-K, 10-Q) — they have no LINKED_TO
    edges, so without SIMILAR_TO Graph RAG would reduce to flat vector search on
    pipeline content.

    PER-ANCHOR CAP (top_n_per_anchor=3):
      Previous design used a single LIMIT across all anchors — 18 sections competed
      for 5 total slots, so most anchors contributed nothing. Now each anchor section
      gets up to top_n_per_anchor neighbor sections independently. With 18 anchors
      this yields up to 54 candidates before cross-encoder reranking filters them down.

    The first shadow of each neighbor section is returned as the content representative.
    Cross-encoder reranking (called after this function) handles final selection.
    """
    result = tx.run("""
        UNWIND $section_ids AS sid
        MATCH (sec:Section {id: sid})-[:SIMILAR_TO]-(neighbor:Section)
        WHERE NOT neighbor.id IN $seen
        MATCH (doc:Document)-[:HAS_SECTION]->(neighbor)
        OPTIONAL MATCH (co:Company)-[:HAS_DOCUMENT]->(doc)
        MATCH (neighbor)-[:HAS_SHADOW]->(sh:Shadow)
        WITH sid, neighbor, doc, co, sh
        ORDER BY sid, neighbor.id, sh.id
        WITH sid, neighbor, doc, co, collect(sh)[0] AS sh
        WITH sid, collect({neighbor: neighbor, doc: doc, co: co, sh: sh})[0..$top_n] AS top_neighbors
        UNWIND top_neighbors AS row
        RETURN row.sh.id            AS shadow_id,
               row.sh.text          AS shadow_text,
               row.sh.embedding     AS embedding,
               null                 AS cluster_id,
               null                 AS next_chunk_text,
               row.neighbor.id      AS section_id,
               row.neighbor.title   AS section_title,
               row.neighbor.summary AS section_summary,
               row.doc.id           AS doc_id,
               row.doc.ticker       AS ticker,
               row.doc.form_type    AS form_type,
               row.doc.period       AS period,
               row.co.id            AS company_id,
               null                 AS contains_parent_id,
               null                 AS contains_parent_label
    """, section_ids=section_ids, seen=list(already_seen), top_n=top_n_per_anchor)
    return result.data()


# ===========================================================================
# Step 5G-c — Follow HAS_KEYWORD edges for cross-company thematic linking
# ===========================================================================

def traverse_keyword(tx, shadow_ids: list, already_seen: set, top_keywords_per_anchor: int = 3, top_neighbors_per_keyword: int = 3) -> list:
    """
    For each anchor shadow, find its top keywords (by HAS_KEYWORD score), then
    return other Shadow nodes (not yet seen) that share those keywords.

    HAS_KEYWORD edges built offline by extract_keywords.py (keyBERT extracts top
    phrases from each Shadow) + load_keywords.py (writes edges to Neo4j with score).
    Connects shadows across companies that share the same keyword node.

    PER-ANCHOR CAP (top_keywords_per_anchor=3, top_neighbors_per_keyword=3):
      Previous design collapsed all 18 anchors into 5 global keywords → most anchors
      contributed nothing to keyword traversal. Now each anchor gets its top
      top_keywords_per_anchor keywords independently, and each keyword returns up to
      top_neighbors_per_keyword neighbor shadows. With 18 anchors this yields up to
      18 × 3 × 3 = 162 candidates (heavily deduplicated in practice) before
      cross-encoder reranking filters them down.

    Cross-company thematic linking: surfaces chunks from companies not selected by
    cluster routing if they share specific keyword nodes with the anchor shadows.
    Cross-encoder reranking (called after this function) handles final selection.
    """
    result = tx.run("""
        UNWIND $shadow_ids AS sid
        MATCH (sh:Shadow {id: sid})-[r:HAS_KEYWORD]->(k:Keyword)
        WITH sid, k, r.score AS kw_score
        ORDER BY sid, kw_score DESC
        WITH sid, collect({k: k, score: kw_score})[0..$top_kw] AS top_kws
        UNWIND top_kws AS kw_row
        WITH sid, kw_row.k AS k, kw_row.score AS kw_score
        MATCH (neighbor:Shadow)-[:HAS_KEYWORD]->(k)
        WHERE NOT neighbor.id IN $seen
        MATCH (sec:Section)-[:HAS_SHADOW]->(neighbor)
        MATCH (doc:Document)-[:HAS_SECTION]->(sec)
        OPTIONAL MATCH (co:Company)-[:HAS_DOCUMENT]->(doc)
        WITH sid, neighbor, sec, doc, co, max(kw_score) AS best_kw_score
        ORDER BY sid, best_kw_score DESC
        WITH sid, collect({neighbor: neighbor, sec: sec, doc: doc, co: co})[0..$top_nb] AS top_neighbors
        UNWIND top_neighbors AS row
        RETURN DISTINCT
               row.neighbor.id        AS shadow_id,
               row.neighbor.text      AS shadow_text,
               row.neighbor.embedding AS embedding,
               null                   AS cluster_id,
               null                   AS next_chunk_text,
               row.sec.id             AS section_id,
               row.sec.title          AS section_title,
               row.sec.summary        AS section_summary,
               row.doc.id             AS doc_id,
               row.doc.ticker         AS ticker,
               row.doc.form_type      AS form_type,
               row.doc.period         AS period,
               row.co.id              AS company_id,
               null                   AS contains_parent_id,
               null                   AS contains_parent_label
    """, shadow_ids=shadow_ids, seen=list(already_seen),
         top_kw=top_keywords_per_anchor, top_nb=top_neighbors_per_keyword)
    return result.data()


# ===========================================================================
# Step 6G — Build supporting paths
# ===========================================================================

def build_paths(traversed: list, scores: dict) -> list:
    """
    Build structured path objects from traversal results.
    scores: {shadow_id: cosine_score} from vector search or SIMILAR_TO re-scoring.
    """
    paths = []
    for row in traversed:
        sid = row["shadow_id"]
        node_ids  = [sid, row["section_id"], row["doc_id"]]
        edge_types = ["HAS_SHADOW", "HAS_SECTION"]
        if row.get("company_id"):
            node_ids.append(row["company_id"])
            edge_types.append("HAS_DOCUMENT")
        elif row.get("contains_parent_id"):
            node_ids.append(row["contains_parent_id"])
            edge_types.append("LINKED_TO")
        ticker = row.get("ticker") or "upload"
        paths.append({
            "node_ids":   node_ids,
            "edge_types": edge_types,
            "score":      round(float(scores.get(sid, 0.0)), 4),
            "label":      f"{ticker} {row['form_type']} {row['period']} | {row['section_title']}",
        })
    paths.sort(key=lambda p: p["score"], reverse=True)
    return paths


# ===========================================================================
# LLM synthesis
# ===========================================================================

def synthesize_rag(question: str, shadows: list) -> str:
    """Flat RAG prompt — raw chunk text only, no graph context."""
    if not shadows:
        return "No relevant documents were found in the knowledge graph for this question."
    chunks = "\n\n---\n\n".join(
        f"[{i+1}] SOURCE: {row.get('ticker') or 'upload'} | {row.get('form_type','')} {row.get('period','')} | {row.get('section_title','')}\nEXCERPT:\n{row['text']}"
        for i, row in enumerate(shadows)
        if row.get("text")
    )
    prompt = f"""You are answering a question using retrieved passages.

The answer may require combining information across multiple passages.

Instructions:
1. Identify the most likely company or entity mentioned in the passages.
2. Combine relevant facts across passages that refer to that entity.
3. Do NOT require that the answer appear in a single passage.
4. Only answer "NOT FOUND" if the passages do not collectively support a clear answer.

Be precise and use only the provided passages.

---

Question:
{question}

---

Passages:
{chunks}

---

Reasoning (brief):
- Candidate entities:
- Selected entity:
- Supporting facts (from passages):

---

Final Answer:"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def synthesize_graph_rag(question: str, traversed: list) -> str:
    """
    Structured Graph RAG prompt.
    Each result includes provenance: ticker, form_type, period, section title.
    LLM is instructed to cite the filing and section in its answer.
    """
    if not traversed:
        return "No relevant documents were found in the knowledge graph for this question."
    context_blocks = []
    for i, row in enumerate(traversed):
        ticker = row.get("ticker") or "upload"
        label = f"{ticker} | {row['form_type']} {row['period']} | {row['section_title']}"
        if row.get("contains_parent_id") and not row.get("company_id"):
            label += f" [attached to: {row['contains_parent_id']} via user-defined edge]"
        text = row["shadow_text"] or ""
        section_summary = row.get("section_summary") or ""
        block = (
            f"[{i+1}] SOURCE: {label}\n"
            f"SECTION SUMMARY: {section_summary}\n"
            f"EXCERPT:\n{text}"
        )
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are answering a question using retrieved evidence from multiple documents.

The answer may NOT appear in a single passage. You must combine information across multiple passages when they refer to the same company or event.

Passages are ordered by relevance — earlier passages are more relevant to the question. Weight them accordingly.

Instructions:
1. Identify the company/entity that the earliest, most relevant passages point to.
2. Focus only on passages related to that entity.
3. Combine facts across those passages to answer the question.
4. Do NOT require the full answer to appear in one place.
5. Only say "NOT FOUND" if the evidence, taken together, does not support a clear answer.

Be precise and grounded only in the provided evidence.

---

Question:
{question}

---

Evidence:
{context}

---

Reasoning (brief):
- Candidate entities:
- Selected entity:
- Supporting facts (combined across passages):

---

Final Answer:"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ===========================================================================
# Orchestrator
# ===========================================================================

class RetrievalOrchestrator:

    def retrieve(self, question: str, mode: str = "graph_rag") -> RetrievalResult:
        t_total = time.time()

        # Step 1 — detect metadata filters
        filters = detect_filters(question)

        # Step 2 — embed
        t0 = time.time()
        q_vec = embed(question)
        embed_ms = (time.time() - t0) * 1000

        # ------------------------------------------------------------------
        # RAG mode
        # ------------------------------------------------------------------
        if mode == "rag":
            t0 = time.time()
            with driver.session() as session:
                shadows = session.execute_read(
                    lambda tx: search_shadows(
                        tx, q_vec, filters, VECTOR_TOP_K_RAG, include_text=True
                    )
                )
            search_ms = (time.time() - t0) * 1000

            # Cross-encoder rerank: replaces cosine ordering for the LLM.
            # Cosine retrieves the candidate set; cross-encoder picks the best order.
            t0_re = time.time()
            shadows = rerank(question, shadows, "text")
            rerank_ms = (time.time() - t0_re) * 1000

            t0 = time.time()
            answer = synthesize_rag(question, shadows)
            llm_ms = (time.time() - t0) * 1000

            total_ms = (time.time() - t_total) * 1000
            rag_paths = [
                {
                    "node_ids":   [row["shadow_id"]],
                    "edge_types": [],
                    "score":      round(float(row["score"]), 4),
                    "label":      f"{row.get('ticker') or 'upload'} {row.get('form_type','')} {row.get('period','')} | {row.get('section_title','')}",
                }
                for row in shadows
            ]
            return RetrievalResult(
                answer=answer,
                supporting_paths=rag_paths,
                clusters_used=None,
                tickers_detected=filters["tickers"],
                filters_applied={k: v for k, v in filters.items() if v},
                cache_status="miss",
                mode="rag",
                latency_breakdown={
                    "embed_ms":    round(embed_ms,    1),
                    "search_ms":   round(search_ms,   1),
                    "traverse_ms": 0,
                    "rerank_ms":   round(rerank_ms,   1),
                    "llm_ms":      round(llm_ms,      1),
                    "total_ms":    round(total_ms,    1),
                },
            )

        # ------------------------------------------------------------------
        # Graph RAG mode
        # ------------------------------------------------------------------

        # Step 3G — global search for cluster routing
        t0 = time.time()
        with driver.session() as session:
            seeds = session.execute_read(
                lambda tx: search_shadows(
                    tx, q_vec, filters, VECTOR_TOP_K_GLOBAL, include_text=False
                )
            )
        cluster_ids = get_top_clusters(seeds, CLUSTER_TOP_N)

        # Step 4G — per-cluster scoring: fetch embeddings, score in Python with numpy.
        # Seed-biased sampling (cap CLUSTER_FETCH_CAP) ensures index-identified seeds
        # are never evicted before the numpy re-score step.
        seed_ids_by_cluster: dict = {}
        for s in seeds:
            cid = s.get("cluster_id")
            if cid is not None:
                seed_ids_by_cluster.setdefault(cid, set()).add(s["shadow_id"])

        anchor_shadows = []
        if cluster_ids:
            for cid in cluster_ids:
                sids = seed_ids_by_cluster.get(cid, set())
                with driver.session() as session:
                    candidates = session.execute_read(
                        lambda tx, c=cid, s=sids: fetch_shadows_in_clusters(tx, [c], filters, seed_ids=s)
                    )
                anchor_shadows.extend(score_and_rank(candidates, q_vec, CHUNKS_PER_CLUSTER))
            anchor_shadows.sort(key=lambda x: x["score"], reverse=True)

        # Fallback: if cluster search returned too few results, use global seeds
        if len(anchor_shadows) < MIN_RESULTS_FALLBACK:
            cluster_ids = None
            with driver.session() as session:
                anchor_shadows = session.execute_read(
                    lambda tx: search_shadows(
                        tx, q_vec, filters, VECTOR_TOP_K_RAG, include_text=True
                    )
                )
        search_ms = (time.time() - t0) * 1000

        # Step 5G — traverse up
        t0 = time.time()
        shadow_ids = [r["shadow_id"] for r in anchor_shadows]
        scores     = {r["shadow_id"]: r["score"] for r in anchor_shadows}

        with driver.session() as session:
            traversed = session.execute_read(
                lambda tx: traverse_up(tx, shadow_ids)
            )

        # Step 5G-b — follow SIMILAR_TO edges for actual graph hop
        seen_section_ids = {row["section_id"] for row in traversed}
        anchor_section_ids = list(seen_section_ids)
        with driver.session() as session:
            similar_rows = session.execute_read(
                lambda tx: traverse_similar(tx, anchor_section_ids, seen_section_ids)
            )
        # score SIMILAR_TO rows against the question vector (same as anchor shadows)
        similar_rows = score_and_rank(similar_rows, q_vec, len(similar_rows))
        for row in similar_rows:
            scores[row["shadow_id"]] = row["score"]
        traversed = traversed + similar_rows

        # Step 5G-c — follow HAS_KEYWORD edges for cross-company thematic linking
        seen_shadow_ids = {row["shadow_id"] for row in traversed}
        with driver.session() as session:
            keyword_rows = session.execute_read(
                lambda tx: traverse_keyword(tx, shadow_ids, seen_shadow_ids)
            )
        keyword_rows = score_and_rank(keyword_rows, q_vec, len(keyword_rows))
        for row in keyword_rows:
            scores[row["shadow_id"]] = row["score"]
        traversed = traversed + keyword_rows

        traverse_ms = (time.time() - t0) * 1000

        # Sort traversed by cosine score descending — initial ordering before rerank.
        traversed.sort(key=lambda r: scores.get(r["shadow_id"], 0.0), reverse=True)

        # Cross-encoder rerank: Graph RAG text is in shadow_text field.
        # Replaces cosine ordering; scores dict updated so paths reflect CE scores.
        t0_re = time.time()
        traversed = rerank(question, traversed, "shadow_text")
        rerank_ms = (time.time() - t0_re) * 1000
        for row in traversed:
            scores[row["shadow_id"]] = row["score"]

        # Cap traversed to top 20 after reranking before LLM synthesis.
        # Cross-encoder has already ordered by relevance — keep best 20 only.
        # Prevents context length errors as traversal pool grows with expanded
        # CHUNKS_PER_CLUSTER, traverse_similar, and traverse_keyword.
        traversed = traversed[:20]

        # Step 6G — build paths
        paths = build_paths(traversed, scores)

        # Step 7G — synthesize
        t0 = time.time()
        answer = synthesize_graph_rag(question, traversed)
        llm_ms = (time.time() - t0) * 1000

        total_ms = (time.time() - t_total) * 1000
        return RetrievalResult(
            answer=answer,
            supporting_paths=paths,
            clusters_used=cluster_ids,
            tickers_detected=filters["tickers"],
            filters_applied={k: v for k, v in filters.items() if v},
            cache_status="miss",
            mode="graph_rag",
            latency_breakdown={
                "embed_ms":    round(embed_ms,    1),
                "search_ms":   round(search_ms,   1),
                "traverse_ms": round(traverse_ms, 1),
                "rerank_ms":   round(rerank_ms,   1),
                "llm_ms":      round(llm_ms,      1),
                "total_ms":    round(total_ms,    1),
            },
        )


# ===========================================================================
# Test runner
# ===========================================================================


QUESTIONSSSSSSSSS = [
    # All questions are ticker-free — no company name mentioned.
    # Each targets a specific fact in a real SEC filing (10-K, 10-Q, 8-K, or DEF 14A).

    # ------------------------------------------------------------------
    # GOEV — Chapter 7 liquidation, no reorganization
    # ------------------------------------------------------------------
    "Which electric vehicle startup, after failing to secure additional financing, ceased all operations in early 2025 and transferred control of its assets to an independent administrator for liquidation rather than attempting to restructure as a going concern?",

    # ------------------------------------------------------------------
    # NKLA — hydrogen infrastructure + restructuring commitments (3 variants)
    # ------------------------------------------------------------------
    "Which EV company that previously emphasized hydrogen infrastructure later entered court-supervised restructuring and indicated it would both pursue asset sales and maintain limited support for deployed vehicles?",

    "Which EV company that previously focused on hydrogen technologies later disclosed that it would continue supporting existing customers while pursuing a sale of its assets under court oversight?",

    "Which electric commercial truck company, having previously developed hydrogen-based powertrain technology, later entered court-supervised restructuring proceedings and announced plans to sell its remaining assets while continuing limited service operations for existing fleet customers?",

    "Which electric commercial truck company, having previously developed hydrogen-based powertrain technology, entered formal insolvency proceedings and announced plans to sell its remaining assets while continuing limited service operations for existing fleet customers?",

    # ------------------------------------------------------------------
    # FSR — creditor negotiations → debt acceleration → bankruptcy (3 variants, Q3 kept)
    # ------------------------------------------------------------------
    "Which electric vehicle manufacturer entered bankruptcy protection in mid-2024 after its restructuring negotiations with creditors collapsed, causing its outstanding debt obligations to become immediately due and payable?",

    "Which EV manufacturer disclosed that failed restructuring negotiations with creditors triggered an event of default that accelerated its outstanding debt, leading it to seek bankruptcy protection?",

    "Which EV company disclosed that its inability to reach agreement with creditors caused its debt to be accelerated, ultimately resulting in a bankruptcy filing?",

    # ------------------------------------------------------------------
    # FSR — causal chain (original Q3, kept)
    # ------------------------------------------------------------------
    # needle: FSR 8-K — failed negotiations → debt acceleration → bankruptcy
    "Which manufacturer disclosed in mid-2024 that failed negotiations with creditors triggered an event causing its obligations to become immediately due, and how did that event lead to its subsequent bankruptcy filing?",

    # ------------------------------------------------------------------
    # NKLA SPAC — business combination + vehicle categories (2 variants)
    # ------------------------------------------------------------------
    "In which EV startup’s quarterly filing from late 2020 did the board first conclude that a business combination had eliminated all substantial doubt about the company’s going concern status, and what two types of electric vehicles was it developing at that time?",

    # ------------------------------------------------------------------
    # RIVN — exclusivity agreement (original Q5, kept)
    # ------------------------------------------------------------------
    # needle: RIVN 10-K / DEF14A — exclusivity → right of first refusal → no minimum purchase
    "Which manufacturer entered into a commercial agreement that restricted its ability to sell a specific vehicle type to other customers for a defined period, and how did that agreement change after the exclusivity window expired?",

    # ------------------------------------------------------------------
    # NKLA — going concern language in annual report (original Q9, kept)
    # ------------------------------------------------------------------
    # needle: NKLA 10-K — going concern language prior to Chapter 11
    "Which commercial vehicle manufacturer later entered bankruptcy proceedings, and what language did it use in its most recent annual report to describe uncertainties about its future operations?",
] 

# ===========================================================================
# OLD QUESTIONS — commented out 2026-03-22
# Problem: all name the company explicitly (Nikola, Fisker, Canoo) so the
# ticker filter does the narrowing before cluster routing gets a chance.
# Question vocabulary also closely matches SEC filing language — no semantic
# gap for Graph RAG to bridge.
# ===========================================================================
# QUESTIONS_OLD = [
#     # NKLA — regulatory scrutiny + going concern (compound condition)
#     # needle: NKLA SEC/Hindenburg scrutiny + going concern doubt
#     # answer: Nikola — SEC investigation from Hindenburg short-seller report +
#     #         "substantial doubt" language in subsequent annual filings.
#     # why graph rag fails: compound embedding blends two separate concepts;
#     #                      scrutiny and going-concern chunks live in different
#     #                      clusters; RAG global search surfaces both independently.
#     "Which company disclosed both regulatory or legal scrutiny of its business practices and later expressed substantial doubt about its ability to continue operating?",
#
#     # NKLA — cash trajectory across 2 annual reports (needs 10-K 2021 + 10-K 2022)
#     # needle: NKLA 10-K 2021 + NKLA 10-K 2022
#     # answer: Cash declined from $497.2M (year-end 2021) to $233.4M (year-end 2022).
#     #         Going concern language appeared in both. Net loss grew from $690.4M to $784.2M.
#     # why graph rag wins: no year filter — must traverse NKLA 10-K cluster to surface
#     #                     both documents; RAG may only retrieve one year.
#     "How did Nikola's cash reserves and going concern language change in the two annual reports before it entered bankruptcy?",
#
#     # NKLA — going concern progression across 3 annual reports (10-K 2022 + 2023 + 2024)
#     # needle: NKLA 10-K 2022 + 2023 + 2024
#     # answer: Net loss grew from $784.2M (2022) to $966.3M (2023). Going concern language
#     #         in both. Final filing (2024 10-K) disclosed Chapter 11 filing; Plan of
#     #         Liquidation confirmed September 12, 2025.
#     # why graph rag wins: no year filter — must traverse all NKLA 10-K nodes; RAG
#     #                     retrieves whichever year ranks highest by cosine.
#     "How did Nikola's financial condition deteriorate across its annual reports leading up to its bankruptcy filing?",
#
#     # FSR — distress escalation across 6 8-K filings (Mar–May 2024)
#     # needle: FSR 8-K 2024-03-18 + 2024-03-25 + 2024-04-03 + 2024-04-04 + 2024-04-22 + 2024-05-08
#     # answer: (1) Mar 18 — 10-K filing failure triggered debt acceleration;
#     #         (2) Mar 25 — NYSE suspended trading ("abnormally low" share price);
#     #         (3) Apr 3  — director McDermott resigned, restructuring explored;
#     #         (4) Apr 4  — forbearance agreement signed;
#     #         (5) Apr 22 — CRO Michael Healy appointed, forbearance extended;
#     #         (6) May 7  — Austrian subsidiary Fisker GmbH filed under Austrian law.
#     # why graph rag wins: form_type=8-K filter + FSR cluster surfaces all six events;
#     #                     RAG with 18 chunks retrieves 1-2 events at most.
#     "What sequence of events did Fisker disclose in its 8-K filings in early 2024 that escalated its financial distress toward bankruptcy?",
#
#     # NKLA — proxy + financial filings cross-form (DEF 14A Apr 2024 + 10-K 2023)
#     # needle: NKLA DEF 14A 2024-04-24 + NKLA 10-K 2023
#     # answer: Proxy proposed (1) reverse stock split to maintain Nasdaq bid price and
#     #         (2) broadening investor base. Concurrent 10-K disclosed going concern doubt
#     #         and net loss of $966.3M.
#     # why graph rag wins: no form_type filter — must retrieve both DEF 14A and 10-K from
#     #                     NKLA cluster; cosine query tends to surface one form type only.
#     "What proposals were put to Nikola stockholders in its April 2024 proxy statement, and what going concern disclosures appeared in Nikola's filings around that same time?",
#
#     # GOEV — business combination closing + first going concern (8-K + 10-Q)
#     # needle: GOEV 8-K Dec 2020 (Hennessy Capital IV merger closed) +
#     #         GOEV 10-Q May 2021 (first going concern disclosure, ~5 months later)
#     # answer: Merger closed December 2020. First quarterly filing (May 2021 10-Q)
#     #         disclosed going concern uncertainty — ~5 months after closing.
#     # why graph rag wins: requires connecting merger 8-K to subsequent 10-Q across
#     #                     form types and periods; no single chunk contains both facts.
#     "What did Canoo disclose in its first quarterly report after completing its business combination, and how soon after the merger did it first raise going concern uncertainty?",
# ]


QUESTIONS = [
    # ==================================================================
    # NEW QUESTIONS — 2026-03-22
    # Design principles:
    #   1. No company name or ticker in the question text.
    #   2. Vocabulary gap: question words differ from chunk words.
    #   3. Multi-document: correct answer requires connecting 2+ filings.
    #   4. Cross-company where possible: requires ranking across corpus.
    # ==================================================================

    # ------------------------------------------------------------------
    # FSR — NYSE suspension + forbearance (vocabulary gap)
    # ------------------------------------------------------------------
    # needle: FSR 8-K 2024-03-25 (NYSE suspended trading, "abnormally low" share price)
    #         + FSR 8-K 2024-04-04 (forbearance agreement signed with creditor)
    # answer: Fisker Inc. — NYSE suspended trading citing abnormally low share price
    #         while the company was simultaneously negotiating a forbearance agreement.
    # vocabulary gap: question says "suspended from trading" / "negotiating a forbearance"
    #                 chunks say "NYSE immediately suspended" / "forbearance agreement" —
    #                 close but spread across two separate 8-K filings; no chunk has both.
    # why graph rag wins: two 8-K filings in different periods must be connected via FSR
    #                     cluster traversal; RAG retrieves one or the other, not both.
    "Which EV company's stock was suspended from trading by a national exchange while it was simultaneously negotiating a forbearance agreement with its creditors?",

    # ------------------------------------------------------------------
    # NKLA — Hindenburg allegations (strong vocabulary gap)
    # ------------------------------------------------------------------
    # needle: NKLA 10-K 2021 / 10-Q (SEC investigation section, Hindenburg Research report)
    # answer: Nikola Corporation — Hindenburg Research published a report alleging the
    #         company misrepresented its technology (including staging a truck rolling
    #         downhill as "in motion under its own power"). Led to SEC/DOJ investigation.
    # vocabulary gap: question says "misrepresented the capabilities of its products
    #                 before they existed" — chunks say "short seller report",
    #                 "Hindenburg Research", "allegations", "SEC subpoena". Real gap.
    # why graph rag wins: allegation chunks and SEC investigation chunks live in different
    #                     sections; cluster traversal connects them; RAG may surface only one.
    "Which EV startup faced public allegations that it had misrepresented the capabilities of its products before those products existed, and which external party first made those allegations?",

    # ------------------------------------------------------------------
    # RIVN — Amazon exclusivity agreement (vocabulary gap)
    # ------------------------------------------------------------------
    # needle: RIVN 10-K 2022 / 2023 (commercial agreement with Amazon, exclusivity period,
    #         right of first refusal, no minimum purchase commitment)
    # answer: Rivian — commercial agreement with Amazon gave Amazon exclusivity on
    #         electric delivery vans through 2025; after expiry converts to right of
    #         first refusal; Amazon has no minimum purchase obligation.
    # vocabulary gap: question says "largest commercial customer" / "exclusive rights" —
    #                 chunks say "commercial agreement", "exclusivity period",
    #                 "right of first refusal". Moderate gap.
    # why graph rag wins: exclusivity terms and right-of-first-refusal terms appear in
    #                     different sections/filings; cluster traversal surfaces both.
    "Which EV manufacturer disclosed that its largest commercial customer held exclusive rights to purchase a specific vehicle type for a defined period, and what happened to those rights after the exclusivity window expired?",

    # ------------------------------------------------------------------
    # GOEV — merger → going concern within 6 months (multi-hop temporal)
    # ------------------------------------------------------------------
    # needle: GOEV 8-K Dec 2020 (Hennessy Capital IV business combination closed) +
    #         GOEV 10-Q May 2021 (first going concern disclosure, ~5 months post-merger)
    # answer: Canoo — business combination with Hennessy Capital IV closed December 2020;
    #         going concern doubt first disclosed in May 2021 10-Q, ~5 months later.
    # vocabulary gap: question says "doubt about its ability to survive" / "supposed to
    #                 fund its operations" — chunks say "going concern" / "substantial doubt"
    #                 / "business combination". Gap on "survive" vs "going concern".
    # why graph rag wins: merger date (8-K) and going concern (10-Q) are in different
    #                     documents with different form_types and periods; no single chunk
    #                     contains both; cluster traversal connects them.
    "Which EV company first raised doubt about its ability to survive within six months of completing a merger that was supposed to provide the capital to fund its operations?",

    # ------------------------------------------------------------------
    # Cross-company — highest cash before bankruptcy (comparative, no ticker)
    # ------------------------------------------------------------------
    # needle: NKLA 10-K 2022 ($233.4M cash at year-end) vs FSR 10-K 2023 vs GOEV 10-K/10-Q
    # answer: Nikola — held ~$233M cash at end of fiscal 2022 (its last full-year filing
    #         before Chapter 11 in August 2023), more than FSR or GOEV at equivalent points.
    # vocabulary gap: question says "held the most cash" — chunks say "cash and cash
    #                 equivalents", "liquidity", "balance sheet". Moderate gap.
    # why graph rag wins: requires retrieving and comparing financial data across NKLA,
    #                     FSR, and GOEV simultaneously; no metadata filter narrows to one
    #                     company; cluster routing must surface all three bankrupt EV clusters.
    "Among the EV startups that later filed for bankruptcy, which one held the most cash at the end of its last complete fiscal year before filing?",

    # ------------------------------------------------------------------
    # Cross-company — which two disclosed going concern in same quarter (no ticker)
    # ------------------------------------------------------------------
    # needle: NKLA 10-Q Q3 2022 + GOEV 10-Q Q3 2022 (both disclosed going concern doubt
    #         in filings for the quarter ending September 30, 2022)
    # answer: Nikola and Canoo both disclosed going concern doubt in their Q3 2022
    #         quarterly reports (period ending September 30, 2022).
    # vocabulary gap: question says "financial survival concerns" — chunks say
    #                 "substantial doubt", "going concern", "ability to continue". Gap exists.
    # why graph rag wins: requires finding two different companies' 10-Q filings from the
    #                     same period; ticker filter cannot help (no company named);
    #                     cluster routing must surface both NKLA and GOEV going-concern clusters.
    "Which two EV startups both disclosed financial survival concerns in their quarterly filings for the same calendar quarter?",
]


if __name__ == "__main__":
    orchestrator = RetrievalOrchestrator()
    run_log = []

    for q in QUESTIONS:
        print(f"\n{'='*65}")
        print(f"Q: {q}")

        for mode in ("graph_rag", "rag"):
            print(f"\n  --- {mode.upper()} ---")
            result = orchestrator.retrieve(q, mode=mode)
            result.print_summary()
            print(f"\n  ANSWER:\n{result.answer}")
            run_log.append(result.to_dict(q))

    os.makedirs("results", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(_ROOT, "results", f"results_{ts}.json")
    with open(output_path, "w") as f:
        json.dump(run_log, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    driver.close()
