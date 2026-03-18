"""
summarize.py
============
Shared utility: generate a short summary for any document.

WHY THIS IS SHARED (not inside ingest.py):
  Summary generation happens in two contexts:
    1. During pipeline ingest (ingest.py) — batch processing of downloaded filings
    2. At runtime when a user uploads a new document via the API
  Keeping the logic here means both use the same prompt and model config,
  so summaries are consistent regardless of how a document enters the system.

WHAT IT PRODUCES:
  A 2-3 sentence plain-English summary of what the document is and what it covers.
  Stored in the `summary` field of the Document node.
  Also used as the `text` field that gets embedded for coarse-level vector search.

WHY gpt-4o-mini:
  Fast, cheap, and more than capable for summarizing financial documents.
  We only send the first ~3,000 tokens of the document, not the whole thing.

Requirements:
  pip install openai python-dotenv
"""

import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_cfg    = yaml.safe_load(open(os.path.join(_ROOT, "config.yaml")))["summarization"]

MODEL        = _cfg["model"]
INPUT_TOKENS = _cfg["input_tokens"]

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Tokenizer for truncation (reuse tiktoken already used in chunking)
# ---------------------------------------------------------------------------

import tiktoken
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _truncate(text: str, max_tokens: int) -> str:
    """Return at most max_tokens tokens of text, decoded back to string."""
    tokens = _tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _tokenizer.decode(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_summary(text: str, meta: dict) -> str:
    """
    Generate a 2-3 sentence summary of a document.

    Args:
        text: The full cleaned document text.
        meta: Dict with keys: ticker, form_type, period (used to set context).

    Returns:
        A plain-English summary string, or a fallback description if the API fails.

    WHY WE PASS meta:
      Telling the model what type of document it is produces better summaries.
      "Summarize this text" gives generic output. "Summarize this Tesla 10-K for FY2023"
      tells the model to focus on financial performance, risks, and strategic direction.
    """
    truncated = _truncate(text, INPUT_TOKENS)

    ticker    = meta.get("ticker", "Unknown")
    form_type = meta.get("form_type", "filing")
    period    = meta.get("period", "unknown period")

    prompt = (
        f"You are summarizing a {form_type} SEC filing for {ticker} covering {period}. "
        f"Write a 2-3 sentence summary that captures: what the company does, "
        f"the key financial or operational highlights for this period, and any major risks "
        f"or strategic themes mentioned. Be factual and concise.\n\n"
        f"Document excerpt:\n{truncated}"
    )

    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.2,   # low temperature = consistent, factual output
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Never crash ingest because of a summary failure.
        # Return a minimal fallback so the Document node still has a text field.
        print(f"    [WARN] Summary generation failed for {ticker} {form_type} {period}: {e}")
        return f"{ticker} {form_type} filing for {period}."
