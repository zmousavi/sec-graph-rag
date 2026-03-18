# Cursor-Style Knowledge Graph Retrieval Architecture

## Purpose

This document defines the target architecture and workflow for transforming the current repository into a Cursor-inspired knowledge graph reasoning engine.

The goal is to replicate the core ideas behind Cursor (vector seeding, local exploration, intelligent caching, and reuse of computation), but applied to a Neo4j-based knowledge graph with Redis as the high-speed cache layer.

This specification is written for an AI coding copilot to:

* Analyze the current repository
* Identify architectural gaps
* Refactor and extend services
* Implement clustering, retrieval routing, and path caching
* Introduce structured orchestration logic

---

# System Goals

Given a user prompt, the system must:

1. Rapidly determine which region(s) of the graph are relevant.
2. Restrict traversal to the smallest meaningful subgraph.
3. Cache and reuse graph paths (not just answers).
4. Minimize expensive LLM calls.
5. Produce answers with explicit supporting graph paths.
6. Improve latency and relevance over time.

---

# High-Level Architecture

## Core Components

### 1. API Layer

* Accepts user prompt
* Returns answer + supporting graph paths
* Handles request normalization

### 2. Retrieval Orchestrator (Core Brain)

Responsible for:

* Cache checks
* Embedding calls
* Cluster routing
* Graph traversal strategy
* Reranking logic
* Cache writes

This component must be modular and testable.

### 3. Neo4j (Source of Truth)

Stores:

* Nodes (entities)
* Relationships
* Node embeddings
* Cluster IDs

Provides:

* Vector similarity search (HNSW)
* Constrained traversal
* Graph Data Science algorithms (Louvain)

### 4. Redis (Speed Layer)

Stores:

* Question → Path cache
* Cluster routing hints
* Hot node metadata
* Subgraph summaries

Used for fast lookup and reuse.

### 5. Embedding Service

Generates embeddings for:

* User prompts
* Nodes (offline)
* Optional cluster embeddings
* Optional path embeddings

### 6. Reranker / Lightweight LLM

Used for:

* Scoring candidate paths
* Selecting best reasoning chains
* Optional final synthesis

---

# Data Model Requirements

## Neo4j Node Properties

Each node must include:

* `id` (stable unique ID)
* `text` (summary or description)
* `embedding` (vector)
* `cluster_id` (from Louvain clustering)
* `type` (optional)
* `popularity` or `centrality` (optional)
* `updated_at` (optional)

## Required Indexes

* Vector index on `embedding`
* Index on `cluster_id`
* Optional full-text index on `text`

---

# Offline Processing Requirements

## 1. Node Embedding Generation

All nodes must have embeddings computed and stored.

## 2. Community Detection (Louvain)

Run once (or periodically):

* Assign `cluster_id` to each node
* Persist `cluster_id` property

Clusters are used as coarse retrieval partitions.

---

# Redis Data Structures

## 1. Question Path Cache (Primary Optimization)

Key:

```
q:{qhash}:paths
```

Value:

* List of objects containing:

  * `node_ids`
  * `edge_types`
  * `score`
  * `answer_snippet`

TTL: configurable (hours to days)
Eviction: LFU or LRU

---

## 2. Question Metadata

Key:

```
q:{qhash}:meta
```

Value:

* embedding hash
* cluster candidates
* top seed nodes
* timestamp

---

## 3. Cluster Hot Cache

Key:

```
cluster:{cluster_id}:hot_nodes
```

Value:

* Frequently accessed node IDs
* Optional adjacency summary

---

## 4. Node Hit Counters

Key:

```
node:{id}:hits
```

Used for:

* Promoting hot nodes
* Influencing reranking

---

# Runtime Workflow

## Step 0 — Normalize Prompt

* Canonicalize input
* Generate deterministic `qhash`

---

## Step 1 — Cache Check

### 1A. Exact Question Hit

If `q:{qhash}:paths` exists and valid:

* Return cached answer and paths
* Skip retrieval

### 1B. Semantic Overlap Check (Optional Advanced)

If embedding similarity to previous queries exceeds threshold:

* Reuse overlapping cached paths
* Re-rank quickly

---

## Step 2 — Coarse Routing (Cluster Selection)

1. Embed prompt → `q_vec`
2. Run Neo4j vector search (global)
3. Retrieve top seed nodes (K=20–50)
4. Extract their `cluster_id`s
5. Rank clusters by:

   * Similarity scores
   * Prior hit rate
   * Centrality metrics

Select top N clusters (1–5).

This drastically reduces traversal space.

---

## Step 3 — Cluster-Constrained Retrieval

Within selected clusters:

* Run vector search restricted by `cluster_id`
* Retrieve anchor nodes
* Add bridging candidates
* Add high-centrality nodes

Build candidate node pool.

---

## Step 4 — Controlled Graph Expansion

Perform bounded traversal:

Constraints:

* Hop limit: 2–4
* Branch factor cap
* Edge-type filtering (optional LLM-guided)

Generate candidate paths.

Paths must contain:

* Ordered node list
* Edge labels
* Aggregate relevance score

---

## Step 5 — Path Reranking

Use lightweight model to score:

* Relevance to prompt
* Logical completeness
* Redundancy
* Freshness (if applicable)

Select top N paths.

---

## Step 6 — Answer Synthesis

If paths are strong:

* Generate final answer using only selected paths
* Include path references

If weak:

* Expand search slightly
* Repeat reranking

---

## Step 7 — Cache Write

Store:

* `q:{qhash}:paths`
* `q:{qhash}:meta`
* Increment node hit counters

The system becomes faster over time.

---

# Required Refactors for Existing Repository

The copilot should:

1. Introduce a Retrieval Orchestrator module.
2. Separate embedding logic from traversal logic.
3. Introduce Redis-based caching abstraction layer.
4. Implement cluster-aware retrieval filtering.
5. Ensure all graph traversals are bounded and configurable.
6. Introduce structured path objects (not raw Cypher results).
7. Add scoring abstraction for path ranking.
8. Implement telemetry for:

   * cache hit rate
   * cluster routing accuracy
   * traversal latency
   * LLM call count

---

# Configuration Parameters (Must Be Adjustable)

* `VECTOR_TOP_K`
* `CLUSTER_TOP_K`
* `MAX_HOPS`
* `MAX_BRANCH_FACTOR`
* `PATH_TOP_K`
* `CACHE_TTL`
* `SEMANTIC_OVERLAP_THRESHOLD`

---

# Output Contract

All retrieval responses must include:

```
{
  answer: string,
  supporting_paths: [
    {
      node_ids: [],
      edge_types: [],
      score: number
    }
  ],
  clusters_used: [],
  cache_status: "hit" | "miss" | "partial",
  latency_breakdown: {}
}
```

---

# Long-Term Enhancements (Optional)

* Path-level embeddings
* Subgraph-level caching
* Learned edge-type prediction
* Multi-question decomposition
* Adaptive hop control based on graph density

---

# Design Principles

1. Cache computation, not just answers.
2. Restrict search space early.
3. Prefer structured reasoning over flat retrieval.
4. Avoid global traversal whenever possible.
5. Make performance observable and measurable.
6. Ensure reproducibility of retrieval paths.

---

# End Goal

Transform the repository into a:

"Path-aware, cluster-routed, cache-optimized knowledge graph reasoning engine inspired by Cursor's architecture."

The copilot should now:

* Analyze current architecture
* Identify missing layers
* Propose refactors
* Implement incremental transformation toward this specification

---

# Implementation Rollout Diagram (Plain-English Plan)

Below is a visual representation of the staged rollout plan in simple terms.

```
                ┌────────────────────────────┐
                │ 1. Measure Current System  │
                │ - Speed                    │
                │ - Quality                  │
                │ - Cost                     │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ 2. Add Traffic Controller  │
                │ (Retrieval Orchestrator)   │
                │ - Same behavior initially  │
                │ - No breaking changes      │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ 3. Show Proof Paths        │
                │ - Return answer + paths    │
                │ - Improve trust/debugging  │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ 4. Cluster Graph Offline   │
                │ - Create "neighborhoods"   │
                │ - Store cluster IDs        │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ 5. Search Best Clusters    │
                │ - Route to likely areas    │
                │ - Fallback if low confidence│
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ 6. Rank & Deduplicate      │
                │ - Score candidate paths    │
                │ - Keep best few            │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ 7. Add Fast Memory (Redis) │
                │ - Cache paths              │
                │ - TTL + freshness checks   │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ 8. Similar Question Reuse  │
                │ - Strict safety thresholds │
                │ - Controlled activation    │
                └──────────────┬─────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ 9. Gradual Rollout         │
                │ - Shadow mode              │
                │ - Small % traffic          │
                │ - Monitor metrics          │
                └────────────────────────────┘
```

## Why This Order?

* We measure first to establish a baseline.
* We introduce structure before performance optimizations.
* We add explainability early (proof paths).
* We introduce clustering before aggressive caching.
* We gate reuse logic behind strict safety thresholds.
* Every phase maintains fallback behavior.

This ensures stability, observability, and incremental improvement without destabilizing the system.

---

