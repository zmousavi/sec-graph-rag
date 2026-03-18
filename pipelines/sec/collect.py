"""
step1_collect.py
================
Step 1: Data Collection (plan.txt Section 1 & 4-Step 1)

WHY THIS EXISTS:
  The plan requires "overlapping datasets from competing entities" (TSLA vs F).
  This script downloads the raw HTML filings from SEC EDGAR so that Step 2
  (the Shadow Ingestor) can parse them into Section Nodes and Shadow Chunks.

WHAT IT DOWNLOADS:
  - 10-K (Annual Report) for fiscal years configured in config.yaml
  - 10-Q (Quarterly Report) for the last N quarters (quarterly_n in config.yaml)
  - DEF 14A (Proxy Statement) for the last N filings (def14a_n in config.yaml)
  - 8-K (Current Report) for the last N filings (eightk_n in config.yaml)
  - Companies: configured in config.yaml

WHY HTML AND NOT PDF:
  HTML preserves the "Item" section headers (e.g. <h1>Item 1A: Risk Factors</h1>)
  that Step 2 uses to split the document into Section Nodes. PDFs are flat blobs.

WHY FREE EDGAR API AND NOT sec_api:
  EDGAR's data.sec.gov API is free and requires no paid key. sec_api costs money
  and only fetched the single latest filing, which doesn't serve our multi-year plan.

Output layout:
  sec_data/annual/    → raw HTML for 10-K filings
  sec_data/quarterly/ → raw HTML for 10-Q filings
  sec_txt/annual/     → plain-text extraction for 10-K  (used in Step 2)
  sec_txt/quarterly/  → plain-text extraction for 10-Q  (used in Step 2)

Usage:
  cd scripts/
  python collect.py

Requirements:
  pip install requests beautifulsoup4 lxml python-dotenv
"""

import os
import re
import time
from typing import Optional
from urllib.parse import urljoin

import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load .env so CONTACT_EMAIL is available via os.getenv()
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration  (loaded from config.yaml at project root)
# ---------------------------------------------------------------------------

# Base directory = project root, resolved from this file's location.
# Works regardless of which directory you run the script from.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load config.yaml — add companies/years there, no code changes needed.
with open(os.path.join(_ROOT, "config.yaml"), "r") as _f:
    _cfg = yaml.safe_load(_f)["sec"]

# ticker → CIK mapping (e.g. {"TSLA": "1318605", "F": "0000037996"})
TICKERS = _cfg["companies"]

# Fiscal year-end years to collect 10-K annual reports for.
# A FY2024 10-K is filed in early 2025, so we search the year after.
ANNUAL_YEARS = set(_cfg["annual_years"])

# Number of most-recent 10-Q filings to grab per company.
QUARTERLY_N = _cfg["quarterly_n"]

# Number of most-recent DEF 14A proxy statements to grab per company.
DEF14A_N = _cfg.get("def14a_n", 4)

# Number of most-recent 8-K current reports to grab per company.
EIGHTK_N = _cfg.get("eightk_n", 20)

OUTPUT_DIRS = {
    "html_annual":    os.path.join(_ROOT, "data", "raw", "sec_html", "annual"),
    "html_quarterly": os.path.join(_ROOT, "data", "raw", "sec_html", "quarterly"),
    "html_def14a":    os.path.join(_ROOT, "data", "raw", "sec_html", "def14a"),
    "html_8k":        os.path.join(_ROOT, "data", "raw", "sec_html", "8k"),
    "txt_annual":     os.path.join(_ROOT, "data", "raw", "sec_txt", "annual"),
    "txt_quarterly":  os.path.join(_ROOT, "data", "raw", "sec_txt", "quarterly"),
    "txt_def14a":     os.path.join(_ROOT, "data", "raw", "sec_txt", "def14a"),
    "txt_8k":         os.path.join(_ROOT, "data", "raw", "sec_txt", "8k"),
}

# SEC EDGAR fair-use policy requires a descriptive User-Agent with a real email.
# Without this, EDGAR will block requests. Set CONTACT_EMAIL in your .env file.
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")
if not CONTACT_EMAIL:
    raise SystemExit("ERROR: Set CONTACT_EMAIL in your .env file (required by SEC EDGAR).")

USER_AGENT = f"FinancialTreeGraph/1.0 ({CONTACT_EMAIL})"

# EDGAR asks crawlers to stay under ~10 requests/second. 0.8s delay is safe.
DELAY = 0.8

# lxml is faster than html.parser; fall back to html.parser if lxml not installed.
BS4_PARSER = "lxml"

# ---------------------------------------------------------------------------
# HTTP session with automatic retry on transient errors
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    """
    Build a requests Session with:
    - Proper User-Agent (EDGAR requirement)
    - Auto-retry on 429 (rate limit) and 5xx server errors
    - Exponential backoff so we don't hammer the server
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent":      USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept":          "text/html,application/json,*/*",
    })
    retries = Retry(
        total=5,                                      # max retry attempts
        backoff_factor=1.0,                           # wait 1s, 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],   # retry on these HTTP codes
        allowed_methods=["GET"],
        respect_retry_after_header=True,              # honor EDGAR's Retry-After header
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# ---------------------------------------------------------------------------
# EDGAR Submissions API
# ---------------------------------------------------------------------------

def get_submissions(session: requests.Session, cik: str) -> dict:
    """
    Hit the EDGAR submissions endpoint to get a company's full filing history.
    Returns a dict with a 'filings.recent' key containing parallel arrays of
    form types, dates, accession numbers, and primary document names.

    Example URL: https://data.sec.gov/submissions/CIK0001318605.json
    """
    cik_padded = cik.lstrip("0").zfill(10)  # EDGAR wants exactly 10 digits
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def iter_filings(submissions: dict):
    """
    The EDGAR submissions JSON stores filing data as parallel arrays (not a list of dicts).
    This generator zips them together for easier iteration.
    Yields: (form_type, filed_at_date, accession_number, primary_document_filename)
    """
    recent   = submissions.get("filings", {}).get("recent", {})
    forms    = recent.get("form", [])
    dates    = recent.get("filingDate", [])
    accnums  = recent.get("accessionNumber", [])
    primdocs = recent.get("primaryDocument", [])
    # zip stops at the shortest list, avoiding index errors if arrays differ in length
    for form, date, accnum, primdoc in zip(forms, dates, accnums, primdocs):
        yield form, date, accnum, primdoc

# ---------------------------------------------------------------------------
# Build EDGAR URLs
# ---------------------------------------------------------------------------

def filing_html_url(cik: str, accession_number: str, primary_doc: str) -> str:
    """
    Construct the direct URL to the primary HTML document for a filing.
    EDGAR URL format: /Archives/edgar/data/{cik}/{accession_no_dashes}/{filename}
    """
    cik_plain  = str(int(cik))                    # e.g. "1318605" (no leading zeros)
    acc_nodash = accession_number.replace("-", "") # e.g. "0000950170-24-010987" → "000095017024010987"
    return f"https://www.sec.gov/Archives/edgar/data/{cik_plain}/{acc_nodash}/{primary_doc}"

def filing_index_url(cik: str, accession_number: str) -> str:
    """
    URL to the filing's index page — lists all documents in the submission.
    Used as a fallback when the direct primary_doc URL doesn't work.
    """
    cik_plain  = str(int(cik))
    acc_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_plain}/{acc_nodash}/{accession_number}-index.htm"

# ---------------------------------------------------------------------------
# Fallback: find the HTML (or txt) document from the filing index page
# ---------------------------------------------------------------------------

def resolve_from_index(session: requests.Session, cik: str, accession_number: str) -> tuple[Optional[str], str]:
    """
    When the primary_doc URL doesn't work (e.g. it's an iXBRL viewer wrapper),
    scrape the filing's index page to find the actual document.

    Returns: (url, doc_type) where doc_type is 'html' or 'txt'

    Preference order:
      1. Any .htm/.html file in the Archives (the real filing HTML)
      2. The .txt submission bundle (EDGAR always generates one — last resort)

    We skip /ixviewer/ links because those are JavaScript viewer shells,
    not the raw document content we need.
    """
    index_url = filing_index_url(cik, accession_number)
    try:
        r = session.get(index_url, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, BS4_PARSER)

        cik_plain  = str(int(cik))
        acc_nodash = accession_number.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{cik_plain}/{acc_nodash}/"

        txt_fallback = None  # hold the .txt URL in case no .htm is found

        for a in soup.find_all("a", href=True):
            href = a["href"]
            ext  = href.lower().split("?")[0]  # strip query params before checking extension
            full = urljoin(base, href) if not href.startswith("http") else href

            # Only consider links inside the EDGAR Archives (ignore nav links etc.)
            if "/Archives/edgar/data/" not in full:
                continue
            # Skip the iXBRL viewer shell — it's not the raw document
            if "/ixviewer" in full.lower():
                continue
            # Best case: found an .htm file → return immediately
            if ext.endswith(".htm") or ext.endswith(".html"):
                return full, "html"
            # Keep the first .txt as a backup (EDGAR always has a full submission .txt)
            if ext.endswith(".txt") and txt_fallback is None:
                txt_fallback = full

        if txt_fallback:
            return txt_fallback, "txt"

    except Exception as e:
        print(f"   [WARN] index parse failed: {e}")

    return None, ""  # nothing found

# ---------------------------------------------------------------------------
# Text extraction: HTML → clean plain text
# ---------------------------------------------------------------------------

def html_to_text(html: str) -> str:
    """
    Convert raw filing HTML to clean plain text that Step 2 can chunk.

    - Tables → pipe-delimited rows wrapped in [TABLE]...[/TABLE] markers
      so the chunker knows not to split a table mid-row.
    - Headings → prefixed with ## so section boundaries are visible in the text.
    - Excess whitespace is collapsed.
    """
    soup = BeautifulSoup(html, BS4_PARSER)

    # Convert each HTML table to a readable pipe-delimited block
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n")

    # Mark headings so Step 2 can use them as section split points
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        h.insert_before("\n\n## " + h.get_text(" ", strip=True) + "\n")

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+\n", "\n", text)   # strip trailing spaces on each line
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse 3+ blank lines → 1
    return text

# ---------------------------------------------------------------------------
# Save both HTML and TXT versions of a filing
# ---------------------------------------------------------------------------

def save_pair(html: str, ticker: str, label: str, html_dir: str, txt_dir: str):
    """
    Write two files per filing:
      - .html  → raw HTML (kept for Step 2 section parsing)
      - .txt   → clean text  (kept for Step 2 shadow chunking)
    Label format: e.g. "TSLA_10K_2024" or "F_10Q_2024-10-28"
    """
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(txt_dir,  exist_ok=True)
    html_path = os.path.join(html_dir, f"{ticker}_{label}.html")
    txt_path  = os.path.join(txt_dir,  f"{ticker}_{label}.txt")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(html_to_text(html))
    print(f"   saved → {html_path}")
    print(f"   saved → {txt_path}")

# ---------------------------------------------------------------------------
# Download a single filing (with fallback logic)
# ---------------------------------------------------------------------------

def download_filing(session: requests.Session, cik: str, accession_number: str,
                    primary_doc: str, ticker: str, label: str,
                    html_dir: str, txt_dir: str):
    """
    Download one filing and save HTML + TXT.

    Two-step approach:
      1. Try the direct primaryDocument URL from the submissions JSON.
         This works for most modern (post-2017) iXBRL filings.
      2. If that returns something too small (<5000 chars) or fails,
         scrape the filing index page to find the real document URL.
         As a last resort, grab the raw .txt submission bundle.
    """
    # Skip if already downloaded (allows re-running without re-downloading)
    html_path = os.path.join(html_dir, f"{ticker}_{label}.html")
    if os.path.exists(html_path):
        print(f"   already exists, skipping: {html_path}")
        return

    # --- Attempt 1: direct URL from submissions JSON ---
    url = filing_html_url(cik, accession_number, primary_doc)
    print(f"   fetching: {url}")
    try:
        time.sleep(DELAY)  # be polite to EDGAR
        r = session.get(url, timeout=60)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        print(f"   [WARN] direct fetch failed ({e}), trying index…")
        html = None

    # --- Attempt 2: scrape the filing index if result looks wrong ---
    # A real 10-K is hundreds of KB; <5000 chars means we got a redirect page or shell
    if not html or len(html) < 5000:
        time.sleep(DELAY)
        fallback_url, doc_type = resolve_from_index(session, cik, accession_number)
        if fallback_url:
            print(f"   fallback URL ({doc_type}): {fallback_url}")
            try:
                r2 = session.get(fallback_url, timeout=60)
                r2.raise_for_status()
                html = r2.text
            except Exception as e2:
                print(f"   [ERROR] fallback fetch failed: {e2}")
                return
        else:
            print(f"   [ERROR] could not resolve any document for {ticker} {label}")
            return

    save_pair(html, ticker, label, html_dir, txt_dir)

# ---------------------------------------------------------------------------
# Per-company logic: find which filings match our targets
# ---------------------------------------------------------------------------

def process_ticker(session: requests.Session, ticker: str, cik: str):
    """
    For a given company:
      1. Fetch the full submission history from EDGAR
      2. Find 10-K filings that match ANNUAL_YEARS
      3. Find the QUARTERLY_N most recent 10-Q filings
      4. Find the DEF14A_N most recent DEF 14A proxy statements
      5. Find the EIGHTK_N most recent 8-K current reports
      6. Download each one
    """
    print(f"\n{'='*60}")
    print(f"Processing {ticker} (CIK {cik})")
    print(f"{'='*60}")

    submissions = get_submissions(session, cik)
    time.sleep(DELAY)

    annual_hits    = {}  # { fiscal_year: (accnum, primdoc) }
    quarterly_hits = []  # [ (filed_at, accnum, primdoc), ... ]
    def14a_hits    = []  # [ (filed_at, accnum, primdoc), ... ]
    eightk_hits    = []  # [ (filed_at, accnum, primdoc), ... ]

    for form, filed_at, accnum, primdoc in iter_filings(submissions):
        filed_year = int(filed_at[:4])  # e.g. "2024-01-29" → 2024

        if form == "10-K":
            # A 10-K filed in year Y reports on fiscal year Y-1.
            fiscal_year = filed_year - 1
            if fiscal_year in ANNUAL_YEARS and fiscal_year not in annual_hits:
                annual_hits[fiscal_year] = (accnum, primdoc)

        elif form == "10-Q":
            if len(quarterly_hits) < QUARTERLY_N:
                quarterly_hits.append((filed_at, accnum, primdoc))

        elif form == "DEF 14A":
            if len(def14a_hits) < DEF14A_N:
                def14a_hits.append((filed_at, accnum, primdoc))

        elif form == "8-K":
            if len(eightk_hits) < EIGHTK_N:
                eightk_hits.append((filed_at, accnum, primdoc))

    print(f"\n  10-K targets found: {sorted(annual_hits.keys())}")
    print(f"  10-Q targets found: {len(quarterly_hits)} quarters")
    print(f"  DEF 14A targets found: {len(def14a_hits)}")
    print(f"  8-K targets found: {len(eightk_hits)}")

    # Download annual 10-K filings
    for fiscal_year, (accnum, primdoc) in sorted(annual_hits.items()):
        label = f"10K_{fiscal_year}"
        print(f"\n  → {ticker} 10-K FY{fiscal_year}  ({accnum})")
        download_filing(session, cik, accnum, primdoc, ticker, label,
                        OUTPUT_DIRS["html_annual"], OUTPUT_DIRS["txt_annual"])

    # Download quarterly 10-Q filings
    for filed_at, accnum, primdoc in quarterly_hits:
        label = f"10Q_{filed_at}"
        print(f"\n  → {ticker} 10-Q filed {filed_at}  ({accnum})")
        download_filing(session, cik, accnum, primdoc, ticker, label,
                        OUTPUT_DIRS["html_quarterly"], OUTPUT_DIRS["txt_quarterly"])

    # Download DEF 14A proxy statements
    for filed_at, accnum, primdoc in def14a_hits:
        label = f"DEF14A_{filed_at}"
        print(f"\n  → {ticker} DEF 14A filed {filed_at}  ({accnum})")
        download_filing(session, cik, accnum, primdoc, ticker, label,
                        OUTPUT_DIRS["html_def14a"], OUTPUT_DIRS["txt_def14a"])

    # Download 8-K current reports
    for filed_at, accnum, primdoc in eightk_hits:
        label = f"8K_{filed_at}"
        print(f"\n  → {ticker} 8-K filed {filed_at}  ({accnum})")
        download_filing(session, cik, accnum, primdoc, ticker, label,
                        OUTPUT_DIRS["html_8k"], OUTPUT_DIRS["txt_8k"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    session = make_session()

    for ticker, cik in TICKERS.items():
        try:
            process_ticker(session, ticker, cik)
        except Exception as e:
            print(f"\n[ERROR] {ticker}: {e}")
            import traceback; traceback.print_exc()

    print("\n\nStep 1 complete. Files written to:")
    for k, v in OUTPUT_DIRS.items():
        print(f"  {k:20s} → {v}")


if __name__ == "__main__":
    main()
