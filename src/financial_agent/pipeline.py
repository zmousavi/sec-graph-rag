"""
Pipeline orchestrator for the Financial RAG system.

Runs the full data pipeline by calling scripts in order:
1. Download SEC filings
2. Clean the data
3. Create chunks
4. Generate embeddings
5. Setup vector database

Usage:
    python -m financial_agent.pipeline          # Run full pipeline
    python -m financial_agent.pipeline --stage download
    python -m financial_agent.pipeline --stage clean
    python -m financial_agent.pipeline --stage chunk
    python -m financial_agent.pipeline --stage embed
    python -m financial_agent.pipeline --stage vectordb
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path so we can import from it
REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def run_download():
    """Stage 1: Download SEC filings."""
    print("\n" + "="*50)
    print("STAGE 1: Downloading SEC filings")
    print("="*50 + "\n")

    import importlib
    download_module = importlib.import_module("01_download_filings")
    download_module.main()


def run_clean():
    """Stage 2: Clean the SEC data."""
    print("\n" + "="*50)
    print("STAGE 2: Cleaning SEC data")
    print("="*50 + "\n")

    import importlib
    clean_module = importlib.import_module("02_clean_sec_data")
    clean_module.main()


def run_chunk():
    """Stage 3: Create chunks from cleaned data."""
    print("\n" + "="*50)
    print("STAGE 3: Creating chunks")
    print("="*50 + "\n")

    import importlib
    chunk_module = importlib.import_module("04_create_chunks")
    chunk_module.main()


def run_embed():
    """Stage 4: Generate embeddings for chunks."""
    print("\n" + "="*50)
    print("STAGE 4: Generating embeddings")
    print("="*50 + "\n")

    import importlib
    embed_module = importlib.import_module("05_create_embeddings")
    embed_module.main()


def run_vectordb():
    """Stage 5: Setup the vector database."""
    print("\n" + "="*50)
    print("STAGE 5: Setting up vector database")
    print("="*50 + "\n")

    import importlib
    vectordb_module = importlib.import_module("06_setup_vector_db")
    vectordb_module.main()


def run_full_pipeline():
    """Run all stages in order."""
    print("\n" + "#"*50)
    print("# RUNNING FULL PIPELINE")
    print("#"*50)

    run_download()
    run_clean()
    run_chunk()
    run_embed()
    run_vectordb()

    print("\n" + "#"*50)
    print("# PIPELINE COMPLETE")
    print("#"*50 + "\n")


STAGES = {
    "download": run_download,
    "clean": run_clean,
    "chunk": run_chunk,
    "embed": run_embed,
    "vectordb": run_vectordb,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the Financial RAG data pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        help="Run a specific stage only. If not specified, runs full pipeline."
    )

    args = parser.parse_args()

    if args.stage:
        STAGES[args.stage]()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
