"""
Financial Agent - RAG-powered agent for SEC financial filings.

Usage:
    from financial_agent import FinancialAgent

    agent = FinancialAgent("path/to/vector_db")
    result = agent.query("What are Apple's revenues?")
    print(result["answer"])

    # With GCS support
    from financial_agent import upload_vector_db, download_vector_db
    upload_vector_db("vector_db", "my-bucket")
"""

from .agent import FinancialAgent
from .vector_db import FinancialVectorDB
from .gcs_storage import upload_vector_db, download_vector_db, list_available_dbs

__all__ = [
    "FinancialAgent",
    "FinancialVectorDB",
    "upload_vector_db",
    "download_vector_db",
    "list_available_dbs",
]
__version__ = "0.1.0"
