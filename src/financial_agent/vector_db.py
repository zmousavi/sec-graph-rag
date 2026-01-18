"""
vector_db.py - Vector database for financial document retrieval.

This module provides the FinancialVectorDB class which:
1. Loads a pre-built FAISS index from disk (or downloads from GCS if not found)
2. Searches for similar document chunks using embeddings
3. Filters results by company, section, or document type
4. Supports hybrid search (semantic + keyword matching)
"""

import os
import json
import re
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional


class FinancialVectorDB:
    """
    Vector database for financial document retrieval with filtering.

    This class loads the FAISS index and metadata, then provides search
    functionality with company and section filtering capabilities.

    Usage:
        db = FinancialVectorDB("path/to/vector_db")
        results = db.search(query_embedding, k=5, company_filter="AAPL")

        # Force download from GCS (overwrites local)
        db = FinancialVectorDB("path/to/vector_db", bucket_name="my-bucket", force_download=True)
    """

    def __init__(self, db_path: str, bucket_name: str = None, blob_prefix: str = "vector_db/", force_download: bool = False):
        """
        Initialize the vector database.

        Args:
            db_path: Path to directory containing faiss_index.index and chunks_metadata.json
            bucket_name: GCS bucket name (optional - if provided, can download from GCS)
            blob_prefix: GCS blob prefix (default: "vector_db/")
            force_download: If True, always download from GCS (overwrites local)
        """
        self.db_path = db_path
        self.bucket_name = bucket_name
        self.blob_prefix = blob_prefix
        self.index = None
        self.chunks = None
        self._load_db(force_download=force_download)

    def _load_db(self, force_download: bool = False):
        """Load FAISS index and chunks metadata from disk, downloading from GCS if needed."""
        index_path = os.path.join(self.db_path, "faiss_index.index")
        metadata_path = os.path.join(self.db_path, "chunks_metadata.json")

        local_exists = os.path.exists(index_path) and os.path.exists(metadata_path)

        # Download from GCS if needed
        if self.bucket_name and (force_download or not local_exists):
            self._download_from_gcs()

        # Check again after potential download
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Vector database not found at {self.db_path}")

        # Load FAISS index (the vector search engine)
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Load chunks metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.chunks = data['chunks']

        print(f"Loaded metadata for {len(self.chunks)} chunks")

    def _download_from_gcs(self):
        """Download vector database files from GCS."""
        from financial_agent.gcs_storage import download_vector_db
        print(f"Downloading vector database from gs://{self.bucket_name}/{self.blob_prefix}")
        download_vector_db(self.bucket_name, self.db_path, self.blob_prefix)

    def search(self, query_embedding: np.ndarray, k: int = 5,
               company_filter: Optional[str] = None,
               section_filter: Optional[str] = None,
               document_type_filter: Optional[str] = None,
               query_text: Optional[str] = None,
               keyword_boost: float = 0.1) -> List[Dict[str, Any]]:
        """
        Hybrid search: combines semantic similarity with keyword matching.

        Args:
            query_embedding: Query vector (will be normalized)
            k: Number of TOP results to return
            company_filter: Filter by company (e.g., 'AAPL', 'MSFT')
            section_filter: Filter by section (e.g., 'Business', 'Risk Factors')
            document_type_filter: Filter by document type (e.g., '10-K')
            query_text: Original query text for keyword boosting (optional)
            keyword_boost: How much to boost score per keyword match (default 0.1)

        Returns:
            List of chunk dictionaries with similarity scores, filtered and re-ranked
        """
        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Get more results for hybrid re-ranking
        has_filters = company_filter or section_filter or document_type_filter
        search_k = min(k * 10, self.index.ntotal) if (has_filters or query_text) else k * 2

        # FAISS similarity search
        similarities, indices = self.index.search(query_embedding, search_k)

        # Extract query keywords for boosting
        query_keywords = []
        if query_text:
            stop_words = {'what', 'are', 'the', 'is', 'a', 'an', 'of', 'in', 'for', 'to', 'and',
                         'or', 'how', 'does', 'do', 'their', 'its', 'this', 'that', 'with',
                         'from', 'about', 'say', 'main', 'key', 'company', 'companies'}
            company_names = {'apple', 'microsoft', 'google', 'amazon', 'meta', 'tesla',
                           'nvidia', 'netflix', 'walmart', 'facebook', 'alphabet'}
            phrase_patterns = {
                'revenue': ['net sales', 'total revenue', 'revenue by', 'sales by category',
                           'products and services', 'segment revenue', 'revenue source'],
                'risk': ['risk factor', 'business risk', 'operational risk', 'market risk'],
                'competition': ['competitive', 'competitors', 'market position'],
                'business': ['business model', 'business segment', 'operations'],
            }
            words = re.findall(r'[a-zA-Z]+', query_text.lower())
            base_keywords = [w for w in words
                           if len(w) > 2 and w not in stop_words and w not in company_names]
            query_keywords = list(base_keywords)
            for kw in base_keywords:
                if kw in phrase_patterns:
                    query_keywords.extend(phrase_patterns[kw])

        candidates = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx].copy()
            base_score = float(similarity)

            # Apply metadata filters
            if company_filter and chunk.get('company_ticker') != company_filter:
                continue
            if section_filter and section_filter.lower() not in chunk.get('section_title', '').lower():
                continue
            if document_type_filter and chunk.get('filing_type') != document_type_filter:
                continue

            # Keyword boosting
            boost = 0.0
            if query_keywords:
                chunk_text = (chunk.get('text', '') + ' ' + chunk.get('section_title', '')).lower()
                matches = sum(1 for kw in query_keywords if kw in chunk_text)
                boost = matches * keyword_boost

            chunk['similarity_score'] = base_score
            chunk['keyword_boost'] = boost
            chunk['final_score'] = base_score + boost
            candidates.append(chunk)

        # Re-rank by final_score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # Return top k with rank assigned
        results = candidates[:k]
        for i, chunk in enumerate(results):
            chunk['rank'] = i + 1

        return results

    def get_companies(self) -> List[str]:
        """Get list of available companies in the database."""
        companies = set()
        for chunk in self.chunks:
            if chunk.get('company_ticker'):
                companies.add(chunk['company_ticker'])
        return sorted(list(companies))

    def get_document_types(self) -> List[str]:
        """Get list of available document types in the database."""
        doc_types = set()
        for chunk in self.chunks:
            if chunk.get('filing_type'):
                doc_types.add(chunk['filing_type'])
        return sorted(list(doc_types))

    def get_sections(self, company: Optional[str] = None) -> List[str]:
        """Get list of available sections, optionally filtered by company."""
        sections = set()
        for chunk in self.chunks:
            if company and chunk.get('company_ticker') != company:
                continue
            if chunk.get('section_title'):
                sections.add(chunk['section_title'])
        return sorted(list(sections))

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_chunks': len(self.chunks),
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.index.d,
            'companies': {},
            'sections': {},
            'document_types': {}
        }

        for chunk in self.chunks:
            company = chunk.get('company_ticker', 'Unknown')
            stats['companies'][company] = stats['companies'].get(company, 0) + 1

            section = chunk.get('section_title', 'Unknown')
            stats['sections'][section] = stats['sections'].get(section, 0) + 1

            doc_type = chunk.get('filing_type', 'Unknown')
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1

        return stats
