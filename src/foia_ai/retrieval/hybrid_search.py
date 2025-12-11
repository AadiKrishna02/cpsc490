from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from ..storage.db import get_session
from ..storage.models import Document, Page

LOGGER = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval system combining BM25-style TF-IDF with dense embeddings.
    
    This implements the approach described in your project goals:
    - BM25 for lexical matching (keyword-based)
    - Dense embeddings for semantic similarity
    - Hybrid scoring for best of both worlds
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", cache_dir: Optional[Path] = None):
        self.embedding_model_name = embedding_model
        self.cache_dir = cache_dir or Path("data/retrieval_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[Any] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        
        self.pages: List[Dict] = []
        self.page_texts: List[str] = []
        
        self.tfidf_cache = self.cache_dir / "tfidf_vectorizer.pkl"
        self.tfidf_matrix_cache = self.cache_dir / "tfidf_matrix.pkl"
        self.embeddings_cache = self.cache_dir / "embeddings.pkl"
        self.corpus_cache = self.cache_dir / "corpus.pkl"
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """Build or load the retrieval index from the database."""
        LOGGER.info("Building hybrid retrieval index...")
        
        self._load_corpus()
        
        if not self.page_texts:
            LOGGER.warning("No pages found in database")
            return
        
        self._build_tfidf_index(force_rebuild)
        
        self._build_embedding_index(force_rebuild)
        
        LOGGER.info("Hybrid index built successfully: %d pages indexed", len(self.pages))
    
    def _load_corpus(self) -> None:
        """Load page corpus from database."""
        if self.corpus_cache.exists():
            LOGGER.info("Loading corpus from cache...")
            with open(self.corpus_cache, 'rb') as f:
                data = pickle.load(f)
                self.pages = data['pages']
                self.page_texts = data['page_texts']
            return
        
        LOGGER.info("Loading corpus from database...")
        with get_session() as session:
            pages = session.query(Page).filter(Page.text.isnot(None)).all()
            
            self.pages = []
            self.page_texts = []
            
            for page in pages:
                if not page.text or not page.text.strip():
                    continue
                
                doc = page.document
                page_data = {
                    'page_id': page.id,
                    'document_id': doc.id,
                    'page_no': page.page_no,
                    'text': page.text,
                    'document_title': doc.title,
                    'document_external_id': doc.external_id,
                    'source_name': doc.source.name,
                    'url': doc.url,
                    'extraction_method': 'OCR' if page.ocr_confidence else 'Text',
                    'word_count': len(page.text.split())
                }
                
                self.pages.append(page_data)
                self.page_texts.append(page.text)
        
        with open(self.corpus_cache, 'wb') as f:
            pickle.dump({
                'pages': self.pages,
                'page_texts': self.page_texts
            }, f)
        
        LOGGER.info("Loaded %d pages from database", len(self.pages))
    
    def _build_tfidf_index(self, force_rebuild: bool = False) -> None:
        """Build TF-IDF index for lexical matching."""
        if not force_rebuild and self.tfidf_cache.exists() and self.tfidf_matrix_cache.exists():
            LOGGER.info("Loading TF-IDF index from cache...")
            with open(self.tfidf_cache, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(self.tfidf_matrix_cache, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            return
        
        LOGGER.info("Building TF-IDF index...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better phrase matching
            min_df=1,  # Keep rare terms (important for specialized documents)
            max_df=0.95,  # Remove very common terms
            sublinear_tf=True,  # Use log scaling (similar to BM25)
            norm='l2'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.page_texts)
        
        with open(self.tfidf_cache, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(self.tfidf_matrix_cache, 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        
        LOGGER.info("TF-IDF index built: %d features", len(self.tfidf_vectorizer.vocabulary_))
    
    def _build_embedding_index(self, force_rebuild: bool = False) -> None:
        """Build dense embedding index for semantic matching."""
        if not force_rebuild and self.embeddings_cache.exists():
            LOGGER.info("Loading embeddings from cache...")
            with open(self.embeddings_cache, 'rb') as f:
                self.embeddings = pickle.load(f)
            return
        
        LOGGER.info("Building embedding index with model: %s", self.embedding_model_name)
        
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(self.page_texts), batch_size):
            batch = self.page_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=True)
            embeddings_list.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings_list)
        
        with open(self.embeddings_cache, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        LOGGER.info("Embeddings built: shape %s", self.embeddings.shape)
    
    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict]:
        """
        Hybrid search combining TF-IDF and embedding similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for TF-IDF vs embeddings (0.0 = only embeddings, 1.0 = only TF-IDF)
        
        Returns:
            List of search results with scores and metadata
        """
        if not self.pages:
            LOGGER.warning("Index not built. Call build_index() first.")
            return []
        
        tfidf_scores = self._get_tfidf_scores(query)
        
        embedding_scores = self._get_embedding_scores(query)
        
        hybrid_scores = alpha * tfidf_scores + (1 - alpha) * embedding_scores
        
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if hybrid_scores[idx] > 0:
                page_data = self.pages[idx].copy()
                page_data.update({
                    'score': float(hybrid_scores[idx]),
                    'tfidf_score': float(tfidf_scores[idx]),
                    'embedding_score': float(embedding_scores[idx]),
                    'rank': len(results) + 1
                })
                results.append(page_data)
        
        return results
    
    def _get_tfidf_scores(self, query: str) -> np.ndarray:
        """Get TF-IDF similarity scores for query."""
        query_vector = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        return scores
    
    def _get_embedding_scores(self, query: str) -> np.ndarray:
        """Get embedding similarity scores for query."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        query_embedding = self.embedding_model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings).flatten()
        return scores
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval index."""
        return {
            'total_pages': len(self.pages),
            'total_words': sum(p['word_count'] for p in self.pages),
            'tfidf_features': len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'sources': len(set(p['source_name'] for p in self.pages)),
            'documents': len(set(p['document_id'] for p in self.pages))
        }


def create_retriever() -> HybridRetriever:
    """Factory function to create and initialize a retriever."""
    retriever = HybridRetriever()
    retriever.build_index()
    return retriever
