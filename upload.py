"""
upload.py
=========
Incremental document ingestion — Step 10 of the pipeline.

PURPOSE:
  The batch pipeline (Steps 1-7) processes SEC filings offline and runs full
  Louvain clustering on the entire corpus. upload.py handles the real-world use
  case: a user uploads any document and attaches it to any node in the tree.

  Unlike the batch pipeline, upload.py:
    - Does NOT run Louvain (too expensive per upload, needs all nodes present).
    - Uses nearest-neighbor cluster assignment instead:
        1. Embed the new section's summary.
        2. Find the most similar existing Section in Neo4j (by cosine similarity).
        3. Assign that section's cluster_id to the new section.
        4. Propagate cluster_id down to the new section's shadow nodes.
      This is fast (one vector lookup), content-aware, and requires no GDS.
      A periodic nightly Louvain rerun can re-optimize assignments later.

HOW IT WORKS:
  1. Read the input text file.
  2. Split into sections (fixed-size, ~2000 tokens per section).
  3. Chunk each section into shadow nodes (1000 tokens, 200-token overlap).
  4. LLM-summarize the document and each section.
  5. Embed: document summary, section summaries, shadow chunk texts.
  6. Nearest-neighbor cluster assignment for each section → propagate to shadows.
  7. Write all nodes and edges to Neo4j (MERGE — idempotent).
  8. Write [:LINKED_TO] from linked_to node to new Document (user-drawn edge).
  9. Append new embeddings to the parquet backup.

RELATIONSHIP SEMANTICS:
  [:LINKED_TO]    — user-drawn parent/child link (arbitrary node types).
  [:HAS_SECTION] / [:HAS_SHADOW] / [:NEXT_CHUNK] — pipeline-style internal edges.

Usage:
  python upload.py --file demo_docs/tsla_q4_2023_earnings.txt \\
                   --linked_to TSLA \\
                   --title "TSLA Q4 2023 Earnings Call" \\
                   --ticker TSLA \\
                   --form_type transcript \\
                   --period 2023-Q4

  python upload.py --file demo_docs/rivn_q4_2023_shareholder_letter.txt \\
                   --linked_to RIVN_10K_2023 \\
                   --title "Rivian Q4 2023 Shareholder Letter" \\
                   --ticker RIVN \\
                   --form_type letter \\
                   --period 2023-Q4

  python upload.py --file demo_docs/battery_supply_chain_2023.txt \\
                   --linked_to <section_node_id> \\
                   --title "Battery Supply Chain 2023 Industry Report" \\
                   --ticker "" \\
                   --form_type article \\
                   --period 2023

Requirements:
  pip install openai neo4j python-dotenv pyyaml tiktoken pandas pyarrow scikit-learn numpy
"""

import argparse
import os
import uuid
import yaml
import numpy as np
import pandas as pd
import tiktoken
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT = os.path.abspath(os.path.dirname(__file__))
_cfg  = yaml.safe_load(open(os.path.join(_ROOT, "config.yaml")))

EMB_MODEL  = _cfg["embedding"]["model"]
EMB_DIMS   = _cfg["embedding"]["dimensions"]
EMB_BATCH  = _cfg["embedding"]["batch_size"]
LLM_MODEL  = _cfg["summarization"]["model"]
MAX_TOKENS = _cfg["summarization"]["input_tokens"]

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")

PARQUET_PATH = os.path.join(
    _ROOT, "manifest",
    f"embeddings_{EMB_MODEL}_{EMB_DIMS}d.parquet"
)

# Shadow chunking constants (match the batch pipeline).
CHUNK_TOKENS   = 1000
CHUNK_OVERLAP  = 200   # 20% of 1000
CHUNK_STRIDE   = CHUNK_TOKENS - CHUNK_OVERLAP  # 800

# Section size: how many tokens per section before we start a new one.
SECTION_TOKENS = 2000

_openai    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_tokenizer = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text into sections of at most SECTION_TOKENS tokens each.
    Each section gets a generic title ("Section 1", "Section 2", ...).

    WHY FIXED-SIZE (not header detection):
      User-uploaded documents vary wildly in format — transcripts, articles,
      letters, reports. Header detection heuristics fail on most of them.
      Fixed-size sections are consistent, always produce multiple sections,
      and give the LLM summarizer a reasonable chunk to work with.
    """
    tokens = _tokenizer.encode(text)
    sections = []
    idx = 1
    for i in range(0, len(tokens), SECTION_TOKENS):
        chunk_tokens = tokens[i:i + SECTION_TOKENS]
        if not chunk_tokens:
            break
        chunk_text = _tokenizer.decode(chunk_tokens)
        sections.append((f"Section {idx}", chunk_text))
        idx += 1
    return sections


def chunk_section(text: str) -> list[tuple[str, int]]:
    """
    Split section text into shadow-sized pieces (CHUNK_TOKENS tokens,
    CHUNK_OVERLAP overlap). Returns list of (chunk_text, token_count).
    """
    tokens = _tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), CHUNK_STRIDE):
        chunk_tokens = tokens[i:i + CHUNK_TOKENS]
        if not chunk_tokens:
            break
        chunks.append((_tokenizer.decode(chunk_tokens), len(chunk_tokens)))
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts."""
    response = _openai.embeddings.create(model=EMB_MODEL, input=texts, dimensions=EMB_DIMS)
    return [r.embedding for r in sorted(response.data, key=lambda x: x.index)]


def embed_all(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches of EMB_BATCH."""
    vectors = []
    for i in range(0, len(texts), EMB_BATCH):
        batch = texts[i:i + EMB_BATCH]
        vectors.extend(embed_batch(batch))
        print(f"  embedded {min(i + EMB_BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    return vectors


# ---------------------------------------------------------------------------
# LLM summarization  (prompts match summarize.py)
# ---------------------------------------------------------------------------

def _truncate(text: str, max_tok: int) -> str:
    tokens = _tokenizer.encode(text)
    return text if len(tokens) <= max_tok else _tokenizer.decode(tokens[:max_tok])


def _llm(prompt: str, max_tokens: int) -> str:
    response = _openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def summarize_document(text: str, title: str, ticker: str, form_type: str, period: str) -> str:
    prompt = (
        f"You are summarizing a {form_type} document for {ticker or 'an organization'} "
        f"covering {period}. Title: \"{title}\".\n"
        f"Write a 2-3 sentence summary that captures: what the document is about, "
        f"the key financial or operational highlights, and any major themes. "
        f"Be factual and concise.\n\n"
        f"Document excerpt:\n{_truncate(text, MAX_TOKENS)}"
    )
    try:
        return _llm(prompt, max_tokens=150)
    except Exception as e:
        print(f"  [WARN] doc summarize: {e}")
        return f"{title} ({ticker} {form_type} {period})."


def summarize_section(section_text: str, section_title: str,
                      title: str, ticker: str, form_type: str, period: str) -> str:
    prompt = (
        f"You are summarizing the '{section_title}' portion of a {form_type} document "
        f"titled \"{title}\" ({ticker or 'unknown company'}, {period}).\n"
        f"Write 1-2 sentences capturing the key content of this section. "
        f"Be specific and factual — avoid generic phrases.\n\n"
        f"Section text:\n{_truncate(section_text, MAX_TOKENS)}"
    )
    try:
        return _llm(prompt, max_tokens=100)
    except Exception as e:
        print(f"  [WARN] section summarize: {e}")
        return f"{section_title} of {title}."


# ---------------------------------------------------------------------------
# Nearest-neighbor cluster assignment
# ---------------------------------------------------------------------------

def assign_cluster(section_embedding: list[float], session) -> int | None:
    """
    Find the most similar existing Section node (by cosine similarity) and
    return its cluster_id.

    WHY QUERY NEO4J (not parquet):
      cluster_id is a property on Section nodes in Neo4j, not stored in parquet.
      Querying Neo4j directly avoids a join between two data sources and always
      reflects the latest cluster assignments (including any Louvain reruns).

    Returns None if no existing sections have cluster_ids yet.
    """
    result = session.run("""
        MATCH (s:Section)
        WHERE s.embedding IS NOT NULL AND s.cluster_id IS NOT NULL
        RETURN s.id AS id, s.embedding AS embedding, s.cluster_id AS cluster_id
    """)
    rows = result.data()

    if not rows:
        print("  [WARN] No existing sections with cluster_id — cluster assignment skipped.")
        return None

    existing_embeddings = np.array([r["embedding"] for r in rows], dtype=np.float32)
    q_vec = np.array([section_embedding], dtype=np.float32)
    sims  = cosine_similarity(q_vec, existing_embeddings)[0]
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_cluster = rows[best_idx]["cluster_id"]
    print(f"    → cluster {best_cluster} (sim={best_sim:.4f} to {rows[best_idx]['id'][:60]})")
    return best_cluster


# ---------------------------------------------------------------------------
# Neo4j writes
# ---------------------------------------------------------------------------

def find_company_id(session, ticker: str) -> str | None:
    """Return the Company node id for a given ticker, or None if not found."""
    if not ticker:
        return None
    result = session.run(
        "MATCH (c:Company {ticker: $ticker}) RETURN c.id AS id LIMIT 1",
        ticker=ticker
    )
    row = result.single()
    return row["id"] if row else None


def write_document(session, node: dict):
    session.run("""
        MERGE (d:Document {id: $id})
        SET d.title      = $title,
            d.ticker     = $ticker,
            d.form_type  = $form_type,
            d.period     = $period,
            d.summary    = $summary,
            d.source_file = $source_file,
            d.uploaded   = true,
            d.updated_at = $updated_at
    """, **node)


def write_section(session, node: dict):
    session.run("""
        MERGE (s:Section {id: $id})
        SET s.doc_id        = $doc_id,
            s.title         = $title,
            s.section_index = $section_index,
            s.summary       = $summary,
            s.cluster_id    = $cluster_id,
            s.updated_at    = $updated_at
    """, **node)


def write_shadows(session, rows: list[dict]):
    """Batch-upsert shadow nodes."""
    session.run("""
        UNWIND $rows AS row
        MERGE (sh:Shadow {id: row.id})
        SET sh.doc_id      = row.doc_id,
            sh.section_id  = row.section_id,
            sh.chunk_index = row.chunk_index,
            sh.text        = row.text,
            sh.token_count = row.token_count,
            sh.cluster_id  = row.cluster_id,
            sh.updated_at  = row.updated_at
    """, rows=rows)


def write_embeddings_neo4j(session, label: str, id_vec_pairs: list[tuple]):
    """Write embedding vectors back to nodes."""
    session.run(f"""
        UNWIND $rows AS row
        MATCH (n:{label} {{id: row.id}})
        SET n.embedding = row.embedding
    """, rows=[{"id": id_, "embedding": emb} for id_, emb in id_vec_pairs])


def write_relationships(session, rels: list[dict]):
    """
    Write a list of relationships. Each dict: {from_id, to_id, type}.
    Groups by type to avoid mixing relationship types in one query.
    """
    by_type: dict[str, list] = {}
    for r in rels:
        by_type.setdefault(r["type"], []).append(r)

    for rel_type, group in by_type.items():
        session.run(f"""
            UNWIND $rows AS row
            MATCH (a {{id: row.from_id}})
            MATCH (b {{id: row.to_id}})
            MERGE (a)-[:{rel_type}]->(b)
        """, rows=group)


# ---------------------------------------------------------------------------
# Parquet backup
# ---------------------------------------------------------------------------

def append_parquet(new_rows: list[dict]):
    """
    Append new embedding rows to the parquet backup.
    Each row: {id, label, embedding}.
    If the parquet doesn't exist yet, creates it.
    """
    df_new = pd.DataFrame(new_rows)
    if os.path.exists(PARQUET_PATH):
        df_old = pd.read_parquet(PARQUET_PATH)
        # Drop any rows with matching ids (in case of re-upload)
        df_old = df_old[~df_old["id"].isin(df_new["id"])]
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(PARQUET_PATH, index=False)
    print(f"  Parquet updated → {len(df_out)} total rows in {PARQUET_PATH}")


# ---------------------------------------------------------------------------
# Main upload flow
# ---------------------------------------------------------------------------

def upload(file_path: str, parent_id: str, title: str,
           ticker: str, form_type: str, period: str):

    now = datetime.now(timezone.utc).isoformat()
    uid = uuid.uuid4().hex[:8]  # short suffix to avoid id collisions

    print(f"\n=== upload.py ===")
    print(f"  file:      {file_path}")
    print(f"  parent:    {parent_id}")
    print(f"  title:     {title}")
    print(f"  ticker:    {ticker or '(none)'}")
    print(f"  form_type: {form_type}")
    print(f"  period:    {period}")
    print()

    # 1. Read file
    with open(file_path, encoding="utf-8") as f:
        full_text = f.read()
    print(f"Read {len(full_text):,} chars from {os.path.basename(file_path)}")

    # 2. Split into sections
    raw_sections = split_sections(full_text)
    print(f"Sections: {len(raw_sections)}")

    # 3. Chunk each section into shadows
    # Structure: [(section_title, section_text, [(chunk_text, token_count), ...])]
    section_chunks = [
        (title_s, text_s, chunk_section(text_s))
        for title_s, text_s in raw_sections
    ]

    total_shadows = sum(len(chunks) for _, _, chunks in section_chunks)
    print(f"Shadows:  {total_shadows}")

    # 4. Summarize document
    print("\nSummarizing document...", end=" ", flush=True)
    doc_summary = summarize_document(full_text, title, ticker, form_type, period)
    print("done")

    # 5. Summarize each section
    print("Summarizing sections...")
    section_summaries = []
    for i, (sec_title, sec_text, _) in enumerate(section_chunks):
        print(f"  [{i+1}/{len(section_chunks)}] {sec_title[:60]}...", end=" ", flush=True)
        s = summarize_section(sec_text, sec_title, title, ticker, form_type, period)
        section_summaries.append(s)
        print("done")

    # 6. Build texts to embed in one batch
    # Order: [doc_summary, sec_summary_0, sec_summary_1, ..., shadow_0_0, shadow_0_1, ...]
    texts_to_embed = [doc_summary] + section_summaries
    shadow_texts_flat = []
    for _, _, chunks in section_chunks:
        for chunk_text, _ in chunks:
            shadow_texts_flat.append(chunk_text)
    texts_to_embed += shadow_texts_flat

    print(f"\nEmbedding {len(texts_to_embed)} texts ({1} doc + {len(section_summaries)} sections + {len(shadow_texts_flat)} shadows)...")
    all_vectors = embed_all(texts_to_embed)

    doc_vec       = all_vectors[0]
    sec_vecs      = all_vectors[1:1 + len(section_summaries)]
    shadow_vecs   = all_vectors[1 + len(section_summaries):]

    # 7. Build IDs
    doc_id = f"upload_{ticker}_{uid}" if ticker else f"upload_{uid}"
    source_file = os.path.abspath(file_path)

    section_ids = [f"{doc_id}_sec_{i+1}" for i in range(len(section_chunks))]

    shadow_id_lists = []   # shadow_id_lists[sec_i] = [shadow_ids]
    shadow_idx = 0
    for i, (_, _, chunks) in enumerate(section_chunks):
        ids = [f"{section_ids[i]}_shd_{j+1}" for j in range(len(chunks))]
        shadow_id_lists.append(ids)
        shadow_idx += len(chunks)

    # 8. Nearest-neighbor cluster assignment (needs Neo4j)
    print("\nAssigning clusters...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    section_clusters = []
    with driver.session() as session:
        for i, sec_vec in enumerate(sec_vecs):
            print(f"  Section {i+1}/{len(sec_vecs)}: ", end="", flush=True)
            cid = assign_cluster(sec_vec, session)
            section_clusters.append(cid)

    # 9. Build node dicts
    doc_node = {
        "id":          doc_id,
        "title":       title,
        "ticker":      ticker,
        "form_type":   form_type,
        "period":      period,
        "summary":     doc_summary,
        "source_file": source_file,
        "updated_at":  now,
    }

    section_nodes = []
    for i, (sec_title, sec_text, _) in enumerate(section_chunks):
        section_nodes.append({
            "id":            section_ids[i],
            "doc_id":        doc_id,
            "title":         sec_title,
            "section_index": i + 1,
            "summary":       section_summaries[i],
            "cluster_id":    section_clusters[i],
            "updated_at":    now,
        })

    shadow_nodes = []
    shadow_vec_idx = 0
    for i, (_, _, chunks) in enumerate(section_chunks):
        for j, (chunk_text, token_count) in enumerate(chunks):
            shadow_nodes.append({
                "id":          shadow_id_lists[i][j],
                "doc_id":      doc_id,
                "section_id":  section_ids[i],
                "chunk_index": j + 1,
                "text":        chunk_text,
                "token_count": token_count,
                "cluster_id":  section_clusters[i],
                "updated_at":  now,
            })
            shadow_vec_idx += 1

    # 10. Write to Neo4j
    print("\nWriting to Neo4j...")
    with driver.session() as session:
        # Nodes
        write_document(session, doc_node)
        print(f"  Document: {doc_id}")

        for sn in section_nodes:
            write_section(session, sn)
        print(f"  Sections: {len(section_nodes)}")

        write_shadows(session, shadow_nodes)
        print(f"  Shadows:  {len(shadow_nodes)}")

        # Embeddings
        write_embeddings_neo4j(session, "Document", [(doc_id, doc_vec)])
        write_embeddings_neo4j(session, "Section",
                               list(zip(section_ids, sec_vecs)))
        # Shadows in batches (may be large)
        all_shadow_id_vec = [
            (shadow_id_lists[i][j], shadow_vecs[
                sum(len(shadow_id_lists[k]) for k in range(i)) + j
            ])
            for i in range(len(section_chunks))
            for j in range(len(shadow_id_lists[i]))
        ]
        for start in range(0, len(all_shadow_id_vec), 500):
            batch = all_shadow_id_vec[start:start + 500]
            write_embeddings_neo4j(session, "Shadow", batch)
        print(f"  Embeddings written.")

        # Relationships
        rels = []

        # User-drawn parent → new Document
        rels.append({"from_id": parent_id, "to_id": doc_id, "type": "LINKED_TO"})

        # Internal edges: Document → Section → Shadow → NEXT_CHUNK
        for sec_id in section_ids:
            rels.append({"from_id": doc_id, "to_id": sec_id, "type": "HAS_SECTION"})

        for i, sh_ids in enumerate(shadow_id_lists):
            for sh_id in sh_ids:
                rels.append({"from_id": section_ids[i], "to_id": sh_id, "type": "HAS_SHADOW"})
            # NEXT_CHUNK chain within each section
            for j in range(len(sh_ids) - 1):
                rels.append({"from_id": sh_ids[j], "to_id": sh_ids[j+1], "type": "NEXT_CHUNK"})

        write_relationships(session, rels)
        print(f"  Relationships: {len(rels)}")

    driver.close()

    # 11. Append to parquet backup
    print("\nUpdating parquet backup...")
    parquet_rows = [{"id": doc_id, "label": "Document", "embedding": doc_vec}]
    for i, sec_id in enumerate(section_ids):
        parquet_rows.append({"id": sec_id, "label": "Section", "embedding": sec_vecs[i]})
    for i in range(len(section_chunks)):
        for j, sh_id in enumerate(shadow_id_lists[i]):
            vec_offset = sum(len(shadow_id_lists[k]) for k in range(i)) + j
            parquet_rows.append({"id": sh_id, "label": "Shadow", "embedding": shadow_vecs[vec_offset]})
    append_parquet(parquet_rows)

    # 12. Summary
    print(f"\n=== Done ===")
    print(f"  Document ID:  {doc_id}")
    print(f"  Sections:     {len(section_nodes)}")
    print(f"  Shadows:      {len(shadow_nodes)}")
    print(f"  Cluster IDs:  {list(dict.fromkeys(c for c in section_clusters if c is not None))}")
    print(f"  Parent edge:  {parent_id} --[:LINKED_TO]--> {doc_id}")
    print(f"\nRe-run retrieve.py to query the updated graph.")
    return doc_id


# ---------------------------------------------------------------------------
# TODO: append_to_document(doc_id, file_path)
# ---------------------------------------------------------------------------
# Appends new content to an existing Document node. No new edge is created —
# this is a pure content mutation on the existing node.
#
# How it would work:
#   1. Load the existing Document from Neo4j (get its current summary, source_file).
#   2. Read the new text from file_path.
#   3. Chunk the new text into new Shadow nodes (same 1000-token / 200-overlap logic).
#   4. LLM-regenerate the Document summary (old content + new content combined).
#   5. Re-embed the Document node (summary changed).
#   6. Find the last Shadow node of the existing document (highest chunk_index
#      across all its sections) and chain the first new Shadow via [:NEXT_CHUNK].
#   7. Assign cluster_id to new shadows via nearest-neighbor (same as upload()).
#   8. Write new Shadow nodes + embeddings to Neo4j.
#   9. Update Document.summary + Document.embedding in Neo4j.
#   10. Append new embeddings to parquet backup.
#
# What does NOT change:
#   - The Document node ID stays the same.
#   - Existing Shadow nodes are untouched.
#   - No [:LINKED_TO] edge is created or modified.
#   - Section nodes: for appended content, a new Section is created
#     (e.g. "Appended Section 1") and linked via [:HAS_SECTION].
#     Existing Sections are not re-summarized (too expensive; Document
#     summary re-generation covers the high-level change).
#
# CLI would be:
#   python upload.py --mode append --expand_node <existing_doc_id> --file <new_text.txt>
#
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a document into the Financial Knowledge Graph."
    )
    parser.add_argument("--mode",        default="upload", choices=["upload", "append"],
                        help="'upload' creates a new Document node. 'append' adds content to an existing one.")

    # --- upload mode args ---
    parser.add_argument("--file",        help="Path to the text file to upload or append.")
    parser.add_argument("--linked_to",   help="[upload] ID of the node to attach this document to via [:LINKED_TO].")
    parser.add_argument("--title",       help="[upload] Human-readable title for the new Document node.")
    parser.add_argument("--ticker",      default="", help="[upload] Stock ticker (e.g. TSLA). Leave empty for non-company docs.")
    parser.add_argument("--form_type",   default="", help="[upload] Document type (e.g. transcript, letter, article, 10-K).")
    parser.add_argument("--period",      default="", help="[upload] Period covered (e.g. 2023-Q4, 2023).")

    # --- append mode args ---
    parser.add_argument("--expand_node", help="[append] ID of the existing Document node to expand with new content.")

    args = parser.parse_args()

    if args.mode == "upload":
        if not args.file or not args.linked_to or not args.title:
            parser.error("--mode upload requires --file, --linked_to, and --title.")
        upload(
            file_path  = args.file,
            parent_id  = args.linked_to,
            title      = args.title,
            ticker     = args.ticker,
            form_type  = args.form_type,
            period     = args.period,
        )
    elif args.mode == "append":
        if not args.expand_node or not args.file:
            parser.error("--mode append requires --expand_node and --file.")
        # TODO: call append_to_document(doc_id=args.expand_node, file_path=args.file)
        raise NotImplementedError("append_to_document() is not yet implemented. See the spec comment above.")
