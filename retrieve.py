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

# How many shadows to retrieve in each search pass.
VECTOR_TOP_K_GLOBAL  = 20   # global pass — used for cluster routing
CHUNKS_PER_CLUSTER   = 3    # top chunks kept per cluster (3 × 6 clusters = 18 anchors)
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
    ranked by sum of similarity scores within each cluster. 
    XXXXXXXXXXXXXXXXXX A big cluster might overshadow a smaller one with a few high scoring chunks, rank by sum not optimal XXXXXXXXXXXXXXXXXX
    Clusters with null cluster_id (singletons) are included as their own group.
    """
    cluster_scores: dict = {}
    for row in seeds:
        cid = row.get("cluster_id")
        if cid is None:
            continue
        cluster_scores[cid] = cluster_scores.get(cid, 0.0) + row["score"]
    ranked = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:top_n]]


# ===========================================================================
# Step 4G — Graph RAG: true cluster-scoped search via embedding fetch + numpy
# ===========================================================================

def fetch_shadows_in_clusters(tx, cluster_ids: list, filters: dict) -> list:
    """
    Fetch ALL shadows (with embeddings) from the selected clusters.
    Metadata filters applied here to stay consistent with global search.
    Embeddings are returned so Python can score them against q_vec.
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
    if len(rows) > 1000:
        rows = random.sample(rows, 1000)
    return rows


def score_and_rank(candidates: list, q_vec: list, top_k: int) -> list:
    """
    Compute cosine similarity between q_vec and each candidate's embedding.
    Returns top_k candidates sorted by score descending, with score injected.
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
        score = float(np.dot(q_norm, v_norm))
        scored.append({**row, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


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

def traverse_similar(tx, section_ids: list, already_seen: set, top_n: int = 5) -> list:
    """
    Follow SIMILAR_TO edges from anchor sections to their neighbors.
    This is the actual multi-hop graph traversal — RAG cannot do this.
    Returns traversal rows in the same format as traverse_up so they
    can be merged into the same context.
    """
    result = tx.run("""
        UNWIND $section_ids AS sid
        MATCH (sec:Section {id: sid})-[:SIMILAR_TO]-(neighbor:Section)
        WHERE NOT neighbor.id IN $seen
        MATCH (doc:Document)-[:HAS_SECTION]->(neighbor)
        OPTIONAL MATCH (co:Company)-[:HAS_DOCUMENT]->(doc)
        MATCH (neighbor)-[:HAS_SHADOW]->(sh:Shadow)
        WITH neighbor, doc, co, sh
        ORDER BY neighbor.id, sh.id
        WITH neighbor, doc, co, collect(sh)[0] AS sh
        RETURN sh.id            AS shadow_id,
               sh.text          AS shadow_text,
               sh.embedding     AS embedding,
               null             AS cluster_id,
               null             AS next_chunk_text,
               neighbor.id      AS section_id,
               neighbor.title   AS section_title,
               neighbor.summary AS section_summary,
               doc.id           AS doc_id,
               doc.ticker       AS ticker,
               doc.form_type    AS form_type,
               doc.period       AS period,
               co.id            AS company_id,
               null             AS contains_parent_id,
               null             AS contains_parent_label
        LIMIT $top_n
    """, section_ids=section_ids, seen=list(already_seen), top_n=top_n)
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

Instructions:
1. Identify the most likely company/entity mentioned across the evidence.
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

        # Step 4G — true cluster-scoped search: fetch all embeddings, score in Python
        # Step 4G — per-cluster scoring: top CHUNKS_PER_CLUSTER from each cluster,
        # pooled together. Cap 1000 per cluster before scoring.
        anchor_shadows = []
        if cluster_ids:
            for cid in cluster_ids:
                with driver.session() as session:
                    candidates = session.execute_read(
                        lambda tx, c=cid: fetch_shadows_in_clusters(tx, [c], filters)
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
        traverse_ms = (time.time() - t0) * 1000

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
                "llm_ms":      round(llm_ms,      1),
                "total_ms":    round(total_ms,    1),
            },
        )


# ===========================================================================
# Test runner
# ===========================================================================


QUESTIONS = [
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

    "Which EV startup disclosed in a quarterly filing that a completed business combination resolved prior going concern uncertainty, and what types of electric vehicles was it developing at the time?",

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

    # ------------------------------------------------------------------
    # NKLA — regulatory scrutiny + going concern (original Q12, kept)
    # ------------------------------------------------------------------
    # needle: NKLA — SEC/Hindenburg scrutiny + later going concern doubt
    "Which company disclosed both regulatory or legal scrutiny of its business practices and later expressed substantial doubt about its ability to continue operating?",
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
