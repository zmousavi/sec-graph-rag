# sec-graph-rag

A Graph RAG evaluation system for SEC filings. Builds a Neo4j knowledge graph from real SEC documents (10-K, 10-Q, 8-K, DEF 14A) across 9 EV companies and evaluates whether graph-structured retrieval (cluster routing → graph traversal → cross-encoder reranking) outperforms flat vector search on a set of ticker-free benchmark questions designed with genuine vocabulary gaps.

## What this is

A research and evaluation codebase, not a library. You run scripts to build the graph and evaluate retrieval. The benchmark questions are designed so the answer vocabulary differs from the question vocabulary — forcing the system to bridge semantic gaps rather than match keywords directly.

Current scores: **Graph RAG 3/6 · RAG 3/6** (on ticker-free vocabulary-gap questions — see `QUESTIONS` in `retrieve.py`)

---

## Graph schema

```
Company
  └─[:HAS_DOCUMENT]─► Document         (summary + source_file, no full text in Neo4j)
       └─[:HAS_SECTION]─► Section      (LLM summary + embedding + cluster_id)
            └─[:HAS_SHADOW]─► Shadow   (raw 1000-token chunk + embedding + cluster_id)
                 └─[:NEXT_CHUNK]─► Shadow  (sequential continuity)

Shadow ─[:HAS_KEYWORD]─► Keyword       (cross-company thematic linking)
Section ─[:SIMILAR_TO]─► Section       (pairwise cosine ≥ 0.78, undirected in GDS)
any ─[:LINKED_TO]─► Document           (user-uploaded documents, runtime only)
```

**Node storage:**
- `Shadow` — stores raw chunk text in Neo4j (~4 KB each)
- `Section` / `Document` — store LLM-generated summary only; full text lives on disk (`source_file`) or GCS (`source_url`)
- `Company` — no text, structural anchor only

**Edge sources:**
- `HAS_DOCUMENT`, `HAS_SECTION`, `HAS_SHADOW`, `NEXT_CHUNK` — written by the ingestion pipeline
- `SIMILAR_TO` — written by `cluster.py` (pairwise cosine on Section embeddings)
- `HAS_KEYWORD` — written by `load_keywords.py` (keyBERT extraction via `extract_keywords.py`)
- `LINKED_TO` — written by `upload.py` at runtime when a user attaches a document

---

## Pipeline — run in this order

### Step 1–3: collect, clean, chunk
```bash
python run_all.py --sec
```
Downloads HTML filings from SEC EDGAR for all companies in `config.yaml`, strips HTML/XBRL noise, splits into Section nodes and Shadow (1000-token, 20% overlap) chunks. Writes `manifest/manifest.json`. No API calls — fast and re-runnable.

### Step 4: summarize
```bash
python summarize.py
```
Reads `manifest.json`. For each Document and Section with no summary yet, reads the source file and calls OpenAI to generate a summary. Writes summaries back to `manifest.json`. Idempotent — skips nodes that already have summaries.

### Step 5: load Neo4j
```bash
python load_neo4j.py
```
Reads `manifest.json` and batch-upserts all nodes and relationships into Neo4j using `MERGE` (safe to re-run, never duplicates). Creates vector index on Shadow embeddings (1536 dims, HNSW).

### Step 6: embed
```bash
python embed.py
```
Queries Neo4j for all nodes with `text` or `summary` but no `embedding`. Batch-embeds via OpenAI `text-embedding-3-small` (1536 dims). Writes embeddings back to Neo4j and backs them up to `manifest/embeddings_text-embedding-3-small_1536d.parquet` so Neo4j can be wiped without re-paying for embeddings.

### Step 7: extract keywords
```bash
python extract_keywords.py
```
Fetches all Shadow nodes from Neo4j. Runs keyBERT-style extraction: computes candidate n-gram embeddings, scores each candidate against its chunk embedding, keeps top phrases per chunk. Writes `manifest/keywords.json`. Checkpointed — safe to interrupt and resume.

### Step 8: load keywords
```bash
python load_keywords.py
```
Reads `manifest/keywords.json`. Batch-MERGEs `Keyword` nodes and `HAS_KEYWORD` edges into Neo4j. These edges connect Shadow nodes across companies that share the same keyword phrase — used by `traverse_keyword()` at retrieval time.

**Must run before `cluster.py`** — `cluster.py` reads keyword structure from Neo4j.

### Step 9: cluster
```bash
python cluster.py
```
Assigns `cluster_id` to every Section and Shadow node via four phases:

**Phase 1 — Embedding similarity edges**
Loads Section embeddings from parquet. Computes pairwise cosine similarity across all Section pairs. Writes `[:SIMILAR_TO {score}]` edges for pairs above `SIMILARITY_THRESHOLD = 0.78`. Cached to `manifest/similarity_edges_t0.78_nN.parquet` — skips recompute if file exists.

**Phase 1b — Keyword co-occurrence edges** *(currently disabled)*
Would add `[:SIMILAR_TO]` edges between Section pairs whose Shadow children share ≥ `KEYWORD_EDGE_MIN_SHARED` keywords. Disabled because generic bankruptcy vocabulary ("going concern", "restructuring") causes distressed EV companies (FSR, NKLA) to merge into the same cluster, hurting disambiguation. Re-enable only with high threshold and frequency filtering on keywords.

**Phase 2 — Louvain community detection**
Projects all Section nodes + `SIMILAR_TO` edges (undirected) into a GDS in-memory graph. Runs `gds.louvain.write` → writes `cluster_id` to every Section node. GDS projection is dropped after use.

**Phase 3 — Propagate cluster_id to Shadows**
Each Shadow inherits `cluster_id` from its parent Section via `HAS_SHADOW`. Shadows do not appear in the Louvain projection — they get cluster assignment by propagation.

**Phase 4 — Verification**
Prints cluster size distribution. Warns if one cluster holds >50% of sections (threshold too low) or if singletons dominate (threshold too high). Current: 121 clusters, modularity=0.757.

---

## Retrieval — how it works

### RAG mode (baseline)
```
embed(question)
→ Neo4j HNSW vector search, top-18 Shadow nodes, metadata WHERE filters
→ cross-encoder rerank (ms-marco-MiniLM-L-6-v2)
→ LLM: source label + raw chunk text
```

### Graph RAG mode
```
embed(question)
→ global vector search top-20 (include_text=False, cheap cluster routing pass)
→ get_top_clusters(): rank cluster_ids by mean cosine score → top 6
→ per-cluster fetch: pull all shadows in each cluster (cap 5000, seed-biased)
   → numpy cosine rescore → keep top 6 per cluster = 36 anchor shadows
→ traverse_up(): shadow → Section → Document → Company
   returns section_title, section_summary, ticker, form_type, period per anchor
→ traverse_similar(): from anchor Sections, follow [:SIMILAR_TO] edges (undirected)
   → up to 3 neighbor Sections per anchor = up to 108 additional candidates
→ traverse_keyword(): from anchor Shadows, follow [:HAS_KEYWORD] → Keyword
   → other Shadows sharing that keyword
   → up to 3 keywords per anchor × 3 neighbors per keyword = up to 324 candidates
→ merge all candidates, deduplicate by shadow_id
→ cross-encoder rerank entire pool → keep top 20
→ LLM: source label + section summary + chunk text (richer than RAG)
```

**Key design choices:**
- Cluster routing uses **mean** cosine score (not sum) to avoid large clusters dominating purely by volume
- Seed-biased sampling: shadows identified by global HNSW search are always preserved when cluster fetch exceeds cap — prevents high-scoring seeds from being evicted by random sampling
- `traverse_similar` and `traverse_keyword` both start from the **anchor nodes**, not chained — parallel traversal, not sequential
- Cross-encoder runs on the full traversal pool before LLM — final ranking is by relevance, not cosine

**Both modes use identical:** embedding model, metadata filters (ticker/year/form_type extracted from question text), top-K anchor count. The only structural difference is Graph RAG adds section summary to the LLM prompt and performs graph traversal.

---

## Metadata detection

Questions are parsed for ticker mentions, years, and form types before retrieval:
- Tickers: keyword matching against known company names → canonical ticker (e.g. "Tesla" → "TSLA"). Single-char tickers ("F") use word-boundary regex to avoid false matches.
- Years: regex `\b(202[0-6])\b`
- Form type: keyword matching ("annual" → "10-K", "quarterly" → "10-Q")

Detected values are applied as `WHERE` filters on the Neo4j vector search. **Both RAG and Graph RAG use the same filters** — this isolates the value of graph traversal.

---

## Config

All tunable parameters in `config.yaml` (companies, filing counts) and at the top of each script:

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `SIMILARITY_THRESHOLD` | `cluster.py` | 0.78 | Cosine threshold for SIMILAR_TO edges |
| `KEYWORD_EDGE_MIN_SHARED` | `cluster.py` | 8 | Min shared keywords for co-occurrence edge (Phase 1b disabled) |
| `VECTOR_TOP_K_GLOBAL` | `retrieve.py` | 20 | Seeds for cluster routing pass |
| `CLUSTER_TOP_N` | `retrieve.py` | 6 | Clusters selected per query |
| `CHUNKS_PER_CLUSTER` | `retrieve.py` | 6 | Anchor shadows kept per cluster |
| `CLUSTER_FETCH_CAP` | `retrieve.py` | 5000 | Max shadows fetched per cluster before sampling |

---

## Requirements

- Python 3.11+
- Neo4j 5.x with GDS plugin (for Louvain)
- OpenAI API key (`text-embedding-3-small` + `gpt-4o-mini`)
- `.env` with `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `OPENAI_API_KEY`

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in keys
```

---

## Evaluate

```bash
python retrieve.py                                    # runs QUESTIONS list, saves results/results_{ts}.json
python read_results.py results/results_{ts}.json      # pretty-print answers, paths, latency
```

Results include per-step latency breakdown (`embed_ms`, `search_ms`, `traverse_ms`, `rerank_ms`, `llm_ms`), supporting graph paths with scores, and clusters used.
