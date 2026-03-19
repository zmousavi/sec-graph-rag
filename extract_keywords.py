"""
extract_keywords.py
===================
KeyBERT-style keyword extraction for Shadow (chunk) nodes.

Produces a JSON file: manifest/keywords.json
  {shadow_id: [{text: str, score: float}, ...], ...}

Pipeline position: run AFTER load_neo4j.py, BEFORE load_keywords.py.

HOW IT WORKS:
  1. Fetch all Shadow nodes (id, text, embedding) from Neo4j
  2. Filter out boilerplate chunks (low information density)
  3. Extract candidate n-grams from each chunk text
  4. Collect all unique candidates across the corpus
  5. Batch-embed all unique candidates (one OpenAI call per batch)
     → Checkpointed to manifest/candidate_embeddings_checkpoint.json
       so interrupted runs can resume without re-embedding.
  6. For each chunk: cosine similarity between chunk embedding and
     each candidate embedding → pick top K as keywords
  7. Save results to manifest/keywords.json

WHY SEPARATE FROM NEO4J WRITES:
  Extraction is expensive (API calls). Keeping it separate from Neo4j
  writes means you can inspect results, tune K or filters, and re-run
  without touching the graph. Run load_keywords.py to write to Neo4j.

BOILERPLATE FILTERS (domain-agnostic):
  - Meaningful word ratio < 0.3 (high stopword density)
  - Fewer than 20 unique non-stopword words
  - URL density > 3 tokens matching URL pattern
  - Most common word > 10% of all words (repetition)

Usage:
  python extract_keywords.py                     # all tickers
  python extract_keywords.py --tickers NKLA FSR  # subset for testing
  python extract_keywords.py --no-checkpoint     # ignore saved checkpoint
"""

import os
import re
import json
import time
import yaml
import argparse
import numpy as np
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROOT    = os.path.abspath(os.path.dirname(__file__))
_cfg     = yaml.safe_load(open(os.path.join(_ROOT, "config.yaml")))

EMB_MODEL   = _cfg["embedding"]["model"]
BATCH_SIZE  = _cfg["embedding"]["batch_size"]
OUTPUT_FILE      = os.path.join(_ROOT, "manifest", "keywords.json")
CHECKPOINT_FILE  = os.path.join(_ROOT, "manifest", "candidate_embeddings_checkpoint.json")

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "")

# Top K keywords to keep per chunk.
KEYWORDS_PER_CHUNK = 5

# N-gram range for candidate extraction.
NGRAM_MIN = 1
NGRAM_MAX = 2

# IDF filter: drop candidates that appear in more than this fraction of chunks.
# "electric vehicle" in 80% of chunks → not discriminative → drop.
MAX_DOC_FREQ_RATIO = 0.30

# Minimum document frequency: drop candidates that appear in fewer than this
# fraction of chunks. Scales automatically with corpus size.
# 0.001 = must appear in at least 0.1% of chunks (e.g. 3/3k or 15/15k).
MIN_DOC_FREQ_RATIO = 0.001

# GOAL OF HAS_KEYWORD EDGES:
#   Cross-company thematic linking — connect chunks from different companies
#   that share the same concept (e.g. "creditor default", "going concern",
#   "equity financing"). This adds edges the embedding-based SIMILAR_TO
#   and document hierarchy cannot provide.
#
#   We intentionally DO NOT filter out proper nouns (company names, people,
#   places) with a named-entity filter. Reason: the IDF upper filter
#   (MAX_DOC_FREQ_RATIO) already removes corpus-wide terms like "nikola"
#   or "fisker" that appear in too many chunks to be discriminative.
#   Adding a capitalization heuristic for NER would be an approximation
#   and could incorrectly drop domain terms like "Federal Reserve" or
#   "Model Y". If keyword quality degrades, revisit this tradeoff.

# Boilerplate filter thresholds.
MIN_MEANINGFUL_WORD_RATIO = 0.30   # non-stopwords / total words
MIN_UNIQUE_WORDS          = 20     # unique non-stopword tokens
MAX_URL_TOKENS            = 3      # URL-like tokens before skipping
MAX_TOP_WORD_RATIO        = 0.10   # most common word / total words

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further get got had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll i'm
i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these
they they'd they'll they're they've this those through to too under until
up very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's will with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
inc corp company llc ltd pursuant thereof herein hereof hereby therein
""".split())

# ---------------------------------------------------------------------------
# Boilerplate detection
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r'(https?://|www\.|\.com|\.gov|\.org)', re.I)
WORD_PATTERN = re.compile(r'\b[a-zA-Z]{3,}\b')


def is_boilerplate(text: str) -> bool:
    """Return True if chunk is low-information boilerplate — skip keyword extraction."""
    words = WORD_PATTERN.findall(text.lower())
    if not words:
        return True

    total = len(words)
    meaningful = [w for w in words if w not in STOPWORDS]

    # Filter 1: meaningful word ratio
    if len(meaningful) / total < MIN_MEANINGFUL_WORD_RATIO:
        return True

    # Filter 2: minimum unique meaningful words
    if len(set(meaningful)) < MIN_UNIQUE_WORDS:
        return True

    # Filter 3: URL density
    url_tokens = len(URL_PATTERN.findall(text))
    if url_tokens > MAX_URL_TOKENS:
        return True

    # Filter 4: single word dominates (repetition / header artifacts)
    if meaningful:
        most_common_count = Counter(meaningful).most_common(1)[0][1]
        if most_common_count / total > MAX_TOP_WORD_RATIO:
            return True

    return False


# ---------------------------------------------------------------------------
# N-gram extraction
# ---------------------------------------------------------------------------

def extract_candidates(text: str) -> list[str]:
    """Extract unique n-grams from text, filtering stopwords.

    Bigrams are stored in sorted word order so 'nikola corporation' and
    'corporation nikola' collapse to the same canonical form.
    """
    words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', text.lower())
    candidates = set()
    for n in range(NGRAM_MIN, NGRAM_MAX + 1):
        for i in range(len(words) - n + 1):
            tokens = words[i:i + n]
            # Skip if any token is a stopword
            if not all(t not in STOPWORDS for t in tokens):
                continue
            # Normalize bigrams by sorting tokens → deduplicate reversed pairs
            gram = " ".join(sorted(tokens) if n > 1 else tokens)
            candidates.add(gram)
    return list(candidates)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI. Returns list of embeddings."""
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [r.embedding for r in resp.data]


def embed_all(texts: list[str], use_checkpoint: bool = True) -> dict[str, list[float]]:
    """
    Embed all unique candidate phrases in batches.
    Returns {phrase: embedding}.

    Checkpoints to CHECKPOINT_FILE every 10 batches so interrupted runs
    can resume without re-embedding already-completed phrases.
    """
    unique = list(set(texts))

    # Load checkpoint if it exists
    result: dict[str, list[float]] = {}
    if use_checkpoint and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            result = json.load(f)
        print(f"  Loaded checkpoint: {len(result)} phrases already embedded.")

    remaining = [p for p in unique if p not in result]
    print(f"  Embedding {len(remaining)} remaining candidates "
          f"(of {len(unique)} total) in batches of {BATCH_SIZE}...")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        embs  = embed_batch(batch)
        for phrase, emb in zip(batch, embs):
            result[phrase] = emb
        time.sleep(0.05)  # small rate-limit buffer

        batch_num = i // BATCH_SIZE
        if batch_num % 10 == 0:
            print(f"    {i + len(batch)}/{len(remaining)}")
            # Save checkpoint every 10 batches
            if use_checkpoint:
                os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)

    # Final checkpoint save
    if use_checkpoint:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        print(f"  Checkpoint saved → {CHECKPOINT_FILE}")

    return result


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine(a: list, b: list) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# Neo4j fetch
# ---------------------------------------------------------------------------

def fetch_shadows(tickers: list[str] | None = None) -> list[dict]:
    """Fetch Shadow nodes with text and embedding from Neo4j.

    If tickers is given, only fetch chunks whose id starts with one of
    those tickers (e.g. 'NKLA_', 'FSR_').
    """
    with driver.session() as session:
        if tickers:
            # Shadow IDs are prefixed by ticker, e.g. "NKLA_annual_..."
            ticker_filter = " OR ".join(f"sh.id STARTS WITH '{t}_'" for t in tickers)
            query = f"""
                MATCH (sh:Shadow)
                WHERE sh.text IS NOT NULL AND sh.embedding IS NOT NULL
                  AND ({ticker_filter})
                RETURN sh.id AS shadow_id, sh.text AS text, sh.embedding AS embedding
            """
        else:
            query = """
                MATCH (sh:Shadow)
                WHERE sh.text IS NOT NULL AND sh.embedding IS NOT NULL
                RETURN sh.id AS shadow_id, sh.text AS text, sh.embedding AS embedding
            """
        rows = session.run(query).data()
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KeyBERT-style keyword extraction")
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Limit to specific tickers, e.g. --tickers NKLA FSR"
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Ignore and overwrite any existing embedding checkpoint"
    )
    args = parser.parse_args()

    use_checkpoint = not args.no_checkpoint
    tickers = [t.upper() for t in args.tickers] if args.tickers else None

    print("=" * 60)
    print("KeyBERT-style keyword extraction")
    if tickers:
        print(f"  Tickers: {', '.join(tickers)}")
    if not use_checkpoint:
        print("  Checkpoint: disabled")
    print("=" * 60)

    # 1. Fetch all shadows
    print("\n[1] Fetching Shadow nodes from Neo4j...")
    shadows = fetch_shadows(tickers=tickers)
    print(f"  {len(shadows)} chunks fetched.")

    # 2. Filter boilerplate
    print("\n[2] Filtering boilerplate chunks...")
    valid = [s for s in shadows if not is_boilerplate(s["text"])]
    skipped = len(shadows) - len(valid)
    print(f"  {len(valid)} kept, {skipped} skipped as boilerplate.")

    # 3. Extract candidates per chunk
    print("\n[3] Extracting n-gram candidates...")
    chunk_candidates: dict[str, list[str]] = {}
    doc_freq: Counter = Counter()  # how many chunks each candidate appears in
    for row in valid:
        cands = extract_candidates(row["text"])
        chunk_candidates[row["shadow_id"]] = cands
        doc_freq.update(set(cands))  # count each candidate once per chunk

    total_chunks = len(valid)
    all_candidates_raw = len(doc_freq)

    # IDF filter: keep only candidates within [min_df, max_df] chunk counts
    min_df = max(1, int(total_chunks * MIN_DOC_FREQ_RATIO))
    max_df = max(1, int(total_chunks * MAX_DOC_FREQ_RATIO))
    kept = {phrase for phrase, count in doc_freq.items()
            if min_df <= count <= max_df}
    # Also update chunk_candidates to only keep surviving candidates
    for sid in chunk_candidates:
        chunk_candidates[sid] = [c for c in chunk_candidates[sid] if c in kept]

    print(f"  {all_candidates_raw} raw candidates → {len(kept)} after IDF filter "
          f"(min_df={min_df}, max_df={max_df}/{total_chunks} chunks)")

    # 4. Batch embed all unique candidates
    print("\n[4] Embedding candidates...")
    candidate_embeddings = embed_all(list(kept), use_checkpoint=use_checkpoint)

    # 5. Score candidates per chunk, keep top K
    print(f"\n[5] Scoring candidates per chunk (top {KEYWORDS_PER_CHUNK})...")
    results: dict[str, list[dict]] = {}
    shadow_emb_map = {s["shadow_id"]: s["embedding"] for s in valid}

    for shadow_id, cands in chunk_candidates.items():
        chunk_emb = shadow_emb_map[shadow_id]
        scored = []
        for phrase in cands:
            emb = candidate_embeddings.get(phrase)
            if emb is None:
                continue
            score = cosine(chunk_emb, emb)
            scored.append({"text": phrase, "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        results[shadow_id] = scored[:KEYWORDS_PER_CHUNK]

    print(f"  Done. {len(results)} chunks have keywords.")

    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[6] Saved → {OUTPUT_FILE}")
    print(f"  Next: inspect manifest/keywords.json, then run load_keywords.py")

    driver.close()


if __name__ == "__main__":
    main()
