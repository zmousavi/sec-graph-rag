# Financial Agent

RAG-powered agent for querying SEC financial filings (10-K annual reports).

Ask questions about company finances, risks, business models, and more - with citations from official SEC documents.

## Installation

```bash
pip install financial-agent
```

## Quick Start

### 1. Get a Gemini API Key (free)

Get your free API key at: https://aistudio.google.com/app/apikey

### 2. Set Environment Variable

```bash
export GOOGLE_API_KEY=your-api-key-here
```

### 3. Download Vector Database

**Option A: Download from GCS (recommended)**
```bash
python -m financial_agent.gcs_storage download --bucket bedrock-financial-agent
```

**Option B: Auto-download when using the agent**
```python
from financial_agent import FinancialVectorDB

# Will auto-download from GCS if local doesn't exist
db = FinancialVectorDB("vector_db", bucket_name="bedrock-financial-agent")
```

### 4. Use the Agent

```python
from financial_agent import FinancialAgent

# Initialize with path to vector database
agent = FinancialAgent("path/to/vector_db")

# Ask a question
result = agent.query("What are Apple's main revenue sources?")

# Print the answer
print(result["answer"])

# Print citations
for citation in result["citations"]:
    print(f"[{citation['reference_number']}] {citation['user_friendly_format']}")
```

## Features

- **Natural Language Queries**: Ask questions in plain English
- **Multi-Company Support**: Query data from AAPL, MSFT, GOOGL, AMZN, TSLA, META, NFLX, NVDA, WMT
- **Citations**: Every answer includes references to source documents
- **Smart Retrieval**: Hybrid search combining semantic similarity and keyword matching

## Example Queries

```python
# Single company questions
agent.query("What are Apple's main revenue sources?")
agent.query("What risks does Tesla face?")

# Comparison questions
agent.query("Compare Apple and Microsoft's business models")

# Specific topics
agent.query("What does Amazon say about competition?")
```

## Response Format

```python
result = agent.query("What are Apple's revenues?")

# result contains:
{
    "query": "What are Apple's revenues?",
    "answer": "According to Apple's 2023 annual report...",
    "citations": [...],
    "retrieved_chunks": 5,
    "model_used": "gemini-2.0-flash-exp",
    "timestamp": "2024-01-15T10:30:00"
}
```

## Requirements

- Python 3.9+
- Google Gemini API key (free tier available)
- Pre-built vector database (download from releases)

---

# Building Your Own Vector Database (Advanced)

If you want to build your own vector database with fresh data or different companies, follow these instructions.

## Pipeline Setup

```bash
git clone https://github.com/zmousavi/FinancialAgent.git
cd FinancialAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[pipeline]"
```

## Environment Variables

Create a `.env` file:

```bash
# SEC API (from sec-api.io)
SEC_API_KEY=your_sec_api_key_here
CONTACT_EMAIL=your_email@example.com

# Google Cloud / Vertex AI (for embeddings)
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

## Run the Pipeline

```bash
# Run full pipeline
python -m financial_agent.pipeline

# Or run individual stages
python -m financial_agent.pipeline --stage download
python -m financial_agent.pipeline --stage clean
python -m financial_agent.pipeline --stage chunk
python -m financial_agent.pipeline --stage embed
python -m financial_agent.pipeline --stage vectordb
```

## Upload Vector Database to GCS

After building the vector database, upload it to Google Cloud Storage:

```bash
python -m financial_agent.gcs_storage upload --bucket bedrock-financial-agent
```

## GCS Setup (First Time)

1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
2. Authenticate:
   ```bash
   gcloud auth application-default login
   ```
3. Set your project:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```
4. Create a bucket:
   ```bash
   gsutil mb gs://your-bucket-name
   ```
5. Update `scripts/config.yaml` with your bucket name:
   ```yaml
   gcs:
     bucket_name: "your-bucket-name"
     vector_db_prefix: "vector_db/"
   ```

## Project Structure

```
financial-agent/
├── src/
│   └── financial_agent/        # The pip-installable library
│       ├── __init__.py
│       ├── agent.py            # FinancialAgent class
│       ├── vector_db.py        # Vector database search (supports GCS)
│       ├── pipeline.py         # Pipeline orchestrator
│       └── gcs_storage.py      # GCS upload/download utilities
├── scripts/                    # Data pipeline scripts
│   ├── 01_download_filings.py
│   ├── 02_clean_sec_data.py
│   ├── 03_analyze_documents.py
│   ├── 04_create_chunks.py
│   ├── 05_create_embeddings.py
│   ├── 06_setup_vector_db.py
│   └── config.yaml             # Pipeline configuration
├── vector_db/                  # Pre-built FAISS index
├── pyproject.toml
└── README.md
```

## License

MIT
