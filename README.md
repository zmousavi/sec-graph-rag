# sec-graph-rag

A Graph RAG evaluation system for SEC filings. Builds a Neo4j knowledge graph from real SEC documents (10-K, 10-Q, 8-K, DEF 14A) and evaluates retrieval quality across two pipelines — Graph RAG and standard RAG — against a set of ticker-free benchmark questions.

## What this is

This is a research and evaluation codebase, not a library. You run scripts to build the pipeline and evaluate retrieval. The core question is whether graph-structured retrieval (cluster routing → traversal → cross-encoder reranking) outperforms flat vector search on financial filings.

## Pipeline

```
collect → clean → embed → load_neo4j → cluster → retrieve → evaluate
```

1. **collect** (`pipelines/sec/collect.py`) — download filings from SEC EDGAR
2. **clean** (`pipelines/sec/clean.py`) — parse and chunk filing sections
3. **embed** (`embed.py`) — generate embeddings via OpenAI
4. **load_neo4j** (`load_neo4j.py`) — load Shadow nodes and Document tree into Neo4j
5. **cluster** (`cluster.py`) — build SIMILAR_TO edges via pairwise cosine
6. **extract_keywords** (`extract_keywords.py`) — keyBERT keyword extraction → HAS_KEYWORD edges
7. **retrieve** (`retrieve.py`) — two-stage retrieval: cosine candidate selection → cross-encoder reranking

## Retrieval modes

- **Graph RAG**: embed query → find cluster → traverse SIMILAR_TO/HAS_KEYWORD edges → cross-encoder rerank → Gemini
- **RAG**: embed query → cosine top-k → cross-encoder rerank → Gemini

## Requirements

- Python 3.10+
- Neo4j (local or AuraDB)
- OpenAI API key (embeddings + LLM fallback)
- Gemini API key
- SEC API key (sec-api.io)

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in keys
```

## Evaluation

```bash
python run_all.py        # run both pipelines against all benchmark questions
python read_results.py   # print scored results
```

Current scores: **Graph RAG 11/13 · RAG 11/13**
