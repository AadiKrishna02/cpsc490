#!/usr/bin/env python3
"""
Hybrid Search System: Semantic Search + BM25
Combines vector embeddings with traditional keyword search for optimal retrieval
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sqlite3
from datetime import datetime
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Installing rank-bm25...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25"])
    from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foia_ai.storage.db import get_session
from foia_ai.storage.models import Document, Page, Source


class HybridSearchSystem:
    """Combines semantic search (embeddings) with BM25 keyword search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize hybrid search system
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.model_name = model_name
        self.embedding_model = None
        self.bm25 = None
        self.faiss_index = None
        self.documents = []
        self.document_chunks = []
        self.chunk_metadata = []
        
        self.semantic_weight = 0.6
        self.bm25_weight = 0.4
        
        print(f"Initializing Hybrid Search System with {model_name}")
        
    def load_embedding_model(self):
        """Load sentence transformer model"""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks
    
    def load_documents_from_db(self, limit: int = None, source_filter: str = None) -> List[Dict]:
        """Load documents from database and prepare for indexing"""
        print("Loading documents from database...")
        
        with get_session() as session:
            query = session.query(Document)
            
            if source_filter:
                source = session.query(Source).filter_by(name=source_filter).first()
                if source:
                    query = query.filter_by(source_id=source.id)
            
            if limit:
                documents = query.limit(limit).all()
            else:
                documents = query.all()
            
            print(f"Processing {len(documents):,} documents...")
            
            processed_docs = []
            
            for i, doc in enumerate(documents, 1):
                if i % 500 == 0:
                    print(f"  Processed {i:,}/{len(documents):,} documents...")
                
                pages = session.query(Page).filter_by(document_id=doc.id).all()
                
                if not pages:
                    continue
                
                full_text = "\n\n".join(p.text or "" for p in pages if p.text)
                
                if len(full_text.strip()) < 100:
                    continue
                
                processed_docs.append({
                    'id': doc.external_id,
                    'title': doc.title or f"Document {doc.external_id}",
                    'source': doc.source.name if doc.source else 'Unknown',
                    'text': full_text,
                    'url': doc.url,
                    'page_count': len(pages),
                    'word_count': len(full_text.split())
                })
        
        print(f"Loaded {len(processed_docs):,} documents with text")
        return processed_docs
    
    def build_search_index(
        self,
        documents: List[Dict],
        chunk_size: int = 512,
        encode_batch_size: int = 32,
        chunk_processing_size: int = 512,
    ):
        """Build both semantic and BM25 search indexes"""
        print("Building hybrid search indexes...")
        
        self.documents = documents
        self.document_chunks = []
        self.chunk_metadata = []
        
        self.load_embedding_model()
        
        print("Creating document chunks...")
        
        for doc_idx, doc in enumerate(documents):
            if 'pages' in doc and doc['pages']:
                for page_data in doc['pages']:
                    page_no = page_data['page_no']
                    page_text = page_data['text']
                    page_chunks = self.chunk_text(page_text, chunk_size)
                    
                    for chunk_idx, chunk in enumerate(page_chunks):
                        self.document_chunks.append(chunk)
                        self.chunk_metadata.append({
                            'doc_idx': doc_idx,
                            'chunk_idx': chunk_idx,
                            'doc_id': doc['id'],
                            'title': doc['title'],
                            'source': doc['source'],
                            'url': doc['url'],
                            'page_no': page_no  # Track page number for citations
                        })
            else:
                chunks = self.chunk_text(doc['text'], chunk_size)
                
                for chunk_idx, chunk in enumerate(chunks):
                    self.document_chunks.append(chunk)
                    self.chunk_metadata.append({
                        'doc_idx': doc_idx,
                        'chunk_idx': chunk_idx,
                        'doc_id': doc['id'],
                        'title': doc['title'],
                        'source': doc['source'],
                        'url': doc['url'],
                        'page_no': None  # No page tracking available
                    })
        
        print(f"Created {len(self.document_chunks):,} chunks from {len(documents):,} documents")
        
        print("Building BM25 index...")
        tokenized_chunks = [chunk.lower().split() for chunk in self.document_chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        del tokenized_chunks
        gc.collect()
        
        self._build_semantic_index_streaming(
            self.document_chunks,
            encode_batch_size=encode_batch_size,
            chunk_processing_size=chunk_processing_size,
        )
        
        print(f"Search indexes built successfully!")
        print(f"   - BM25 index: {len(self.document_chunks):,} chunks")
        print(f"   - Semantic index: {self.faiss_index.ntotal:,} vectors ({self.embedding_model.get_sentence_embedding_dimension()}D)")
    
    def search_bm25(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Search using BM25"""
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        
        return results
    
    def _build_semantic_index_streaming(
        self,
        chunk_texts: List[str],
        encode_batch_size: int = 32,
        chunk_processing_size: int = 512,
    ) -> None:
        """
        Build FAISS index by encoding and adding embeddings in small batches.
        
        Args:
            chunk_texts: All chunk texts to embed.
            encode_batch_size: Batch size passed to SentenceTransformer.encode (lower = less RAM).
            chunk_processing_size: Number of chunk texts to encode before adding to FAISS.
        """
        total_chunks = len(chunk_texts)
        if total_chunks == 0:
            print("No chunks available for semantic index")
            self.faiss_index = faiss.IndexFlatIP(1)
            return
        
        print("Building semantic embeddings (streaming mode)...")
        print(f"   Total chunks: {total_chunks:,}")
        print(f"   encode_batch_size (model): {encode_batch_size}")
        print(f"   chunk_processing_size: {chunk_processing_size}")
        
        try:
            import os
            os.nice(10)
        except Exception:
            pass
        
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        processed = 0
        start_time = datetime.now()
        
        for start in range(0, total_chunks, chunk_processing_size):
            batch_texts = chunk_texts[start:start + chunk_processing_size]
            try:
                import torch
                torch.set_num_threads(1)
                
                embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=encode_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    device='cpu'
                ).astype('float32')
                
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    
            except Exception as exc:
                print(f"Error encoding chunk batch starting at {start}: {exc}")
                raise
            
            add_block = 128  # Reduced from 512 to avoid memory spikes
            for block_start in range(0, len(embeddings), add_block):
                block = embeddings[block_start:block_start + add_block]
                block = np.ascontiguousarray(block, dtype='float32')
                faiss.normalize_L2(block)
                self.faiss_index.add(block)
                
                del block
            
            processed += len(batch_texts)
            elapsed = datetime.now() - start_time
            pct = processed / total_chunks * 100
            print(f"     Processed {processed:,}/{total_chunks:,} chunks ({pct:.1f}%) - elapsed {elapsed}")
            
            del embeddings
            del batch_texts
            gc.collect()
            
            if (start // chunk_processing_size) % 10 == 0 and start > 0:
                import time
                time.sleep(0.1)  # Brief pause to let OS reclaim memory
                gc.collect()
        
        print(f"Added {self.faiss_index.ntotal:,} embeddings to FAISS index")
    
    def search_semantic(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Search using semantic embeddings"""
        if not self.faiss_index or not self.embedding_model:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]
        return results
    
    def hybrid_search(self, query: str, top_k: int = 20, 
                     semantic_weight: float = None, bm25_weight: float = None,
                     diversity_mode: str = 'balanced') -> List[Dict]:
        """
        Perform hybrid search combining BM25 and semantic search with diversity control
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (default: 0.6)
            bm25_weight: Weight for BM25 search (default: 0.4)
            diversity_mode: How to handle multiple chunks from same document
                - 'strict': Max 1 chunk per document (maximum diversity)
                - 'balanced': Max 2 chunks per document (default)
                - 'relaxed': Max 3 chunks per document
                - 'best': Take best chunks regardless of source (no diversity)
        """
        if semantic_weight is None:
            semantic_weight = self.semantic_weight
        if bm25_weight is None:
            bm25_weight = self.bm25_weight
        
        print(f"Hybrid search: '{query}' (semantic: {semantic_weight}, BM25: {bm25_weight}, diversity: {diversity_mode})")
        
        bm25_results = self.search_bm25(query, top_k * 5)  # Get more candidates
        semantic_results = self.search_semantic(query, top_k * 5)
        
        def normalize_scores(results):
            if not results:
                return {}
            max_score = max(score for _, score in results)
            min_score = min(score for _, score in results)
            if max_score == min_score:
                return {idx: 1.0 for idx, _ in results}
            return {idx: (score - min_score) / (max_score - min_score) 
                   for idx, score in results}
        
        bm25_normalized = normalize_scores(bm25_results)
        semantic_normalized = normalize_scores(semantic_results)
        
        combined_scores = defaultdict(float)
        
        for idx, score in bm25_normalized.items():
            combined_scores[idx] += bm25_weight * score
        
        for idx, score in semantic_normalized.items():
            combined_scores[idx] += semantic_weight * score
        
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        diversity_limits = {
            'strict': 1,      # 1 chunk per document
            'balanced': 2,    # 2 chunks per document
            'relaxed': 3,     # 3 chunks per document
            'best': 999       # No limit
        }
        max_chunks_per_doc = diversity_limits.get(diversity_mode, 2)
        
        results = []
        doc_chunk_count = defaultdict(int)  # Track chunks per document
        
        for chunk_idx, score in sorted_results:
            if len(results) >= top_k:
                break
                
            metadata = self.chunk_metadata[chunk_idx]
            doc_id = metadata['doc_id']
            
            if doc_chunk_count[doc_id] >= max_chunks_per_doc:
                continue
            
            doc_chunk_count[doc_id] += 1
            
            chunk_text = self.document_chunks[chunk_idx]
            
            results.append({
                'doc_id': doc_id,
                'title': metadata['title'],
                'source': metadata['source'],
                'url': metadata['url'],
                'chunk_text': chunk_text,
                'score': score,
                'bm25_score': bm25_normalized.get(chunk_idx, 0),
                'semantic_score': semantic_normalized.get(chunk_idx, 0),
                'chunk_idx': metadata['chunk_idx'],
                'page_no': metadata.get('page_no')  # Include page number for citations
            })
        
        unique_docs = len(set(r['doc_id'] for r in results))
        avg_chunks_per_doc = len(results) / unique_docs if unique_docs > 0 else 0
        
        print(f"Found {len(results)} chunks from {unique_docs} unique documents (avg: {avg_chunks_per_doc:.1f} chunks/doc)")
        return results
    
    def _convert_indices_to_results(self, raw_results: List[Tuple[int, float]]) -> List[Dict]:
        """
        Convert raw search results (index, score) to full result dictionaries.
        Used by semantic and BM25 search methods.
        """
        results = []
        for chunk_idx, score in raw_results:
            if chunk_idx >= len(self.chunk_metadata):
                continue
                
            metadata = self.chunk_metadata[chunk_idx]
            chunk_text = self.document_chunks[chunk_idx]
            
            results.append({
                'doc_id': metadata['doc_id'],
                'title': metadata['title'],
                'source': metadata['source'],
                'url': metadata['url'],
                'chunk_text': chunk_text,
                'score': score,
                'bm25_score': 0,  # Not available in this context
                'semantic_score': 0,  # Not available in this context
                'chunk_idx': metadata['chunk_idx'],
                'page_no': metadata.get('page_no')  # Include page number
            })
        
        return results
    
    def save_index(self, index_path: str = None):
        """Save the search index to disk"""
        if not index_path:
            timestamp = datetime.now().strftime("%Y%m%D_%H%M%S")
            index_path = f"hybrid_search_index_{timestamp}"
        
        index_dir = ROOT / "data" / "search_indexes" / index_path
        index_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving search index to {index_dir}")
        
        try:
            if self.faiss_index:
                print("  ├─ Saving FAISS semantic index...")
                faiss.write_index(self.faiss_index, str(index_dir / "semantic.faiss"))
                print(f"Saved {self.faiss_index.ntotal:,} vectors")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            raise
        
        try:
            if self.bm25:
                print("  ├─ Saving BM25 keyword index...")
                with open(index_dir / "bm25.pkl", 'wb') as f:
                    pickle.dump(self.bm25, f)
                print("BM25 index saved")
        except Exception as e:
            print(f"Error saving BM25 index: {e}")
            raise
        
        try:
            print("  ├─ Saving document chunks...")
            with open(index_dir / "document_chunks.pkl", 'wb') as f:
                pickle.dump(self.document_chunks, f)
            print(f"Saved {len(self.document_chunks):,} chunks")
        except Exception as e:
            print(f"Error saving document chunks: {e}")
            raise
        
        try:
            print("  ├─ Saving metadata...")
            metadata = {
                'model_name': self.model_name,
                'documents': self.documents,
                'chunk_metadata': self.chunk_metadata,
                'semantic_weight': self.semantic_weight,
                'bm25_weight': self.bm25_weight,
                'created_at': datetime.now().isoformat(),
                'total_documents': len(self.documents),
                'total_chunks': len(self.document_chunks)
            }
            
            with open(index_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            print("Metadata saved")
        except Exception as e:
            print(f"Error saving metadata: {e}")
            raise
        
        print(f"Index saved successfully to {index_dir.name}")
        return index_dir
    
    def load_index(self, index_path: str, shared_embedding_model=None):
        """
        Load a previously saved search index
        
        Args:
            index_path: Path to index directory
            shared_embedding_model: Optional shared SentenceTransformer model to reuse
        """
        index_dir = Path(index_path)
        
        print(f"Loading search index from {index_dir}")
        
        with open(index_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.documents = metadata['documents']
        self.chunk_metadata = metadata['chunk_metadata']
        self.semantic_weight = metadata['semantic_weight']
        self.bm25_weight = metadata['bm25_weight']
        
        if (index_dir / "document_chunks.pkl").exists():
            with open(index_dir / "document_chunks.pkl", 'rb') as f:
                self.document_chunks = pickle.load(f)
        else:
            self.document_chunks = metadata.get('document_chunks', [])
        
        if shared_embedding_model is not None:
            self.embedding_model = shared_embedding_model
        else:
            self.load_embedding_model()
        
        if (index_dir / "semantic.faiss").exists():
            try:
                print("   Using memory-mapped index for efficiency")
                self.faiss_index = faiss.read_index(str(index_dir / "semantic.faiss"), faiss.IO_FLAG_MMAP)
            except Exception as e:
                print(f"MMAP failed ({e}), falling back to full load")
                self.faiss_index = faiss.read_index(str(index_dir / "semantic.faiss"))
        
        if (index_dir / "bm25.pkl").exists():
            with open(index_dir / "bm25.pkl", 'rb') as f:
                self.bm25 = pickle.load(f)
        
        print(f"Index loaded successfully")
        print(f"   - Documents: {len(self.documents):,}")
        print(f"   - Chunks: {len(self.document_chunks):,}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Search System")
    parser.add_argument("--build", action="store_true", help="Build search index")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--load", type=str, help="Load existing index")
    parser.add_argument("--limit", type=int, help="Limit documents for building")
    parser.add_argument("--source", type=str, help="Filter by source")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--semantic-weight", type=float, default=0.6, help="Semantic search weight")
    parser.add_argument("--bm25-weight", type=float, default=0.4, help="BM25 search weight")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Hybrid Search System (Semantic + BM25)")
    print("="*60)
    
    search_system = HybridSearchSystem()
    
    if args.load:
        search_system.load_index(args.load)
    elif args.build:
        documents = search_system.load_documents_from_db(
            limit=args.limit,
            source_filter=args.source
        )
        search_system.build_search_index(documents)
        index_path = search_system.save_index()
        print(f"Index saved to: {index_path}")
    
    if args.search:
        if not search_system.faiss_index or not search_system.bm25:
            print("No search index loaded. Use --build or --load first.")
            return
        
        results = search_system.hybrid_search(
            args.search,
            top_k=args.top_k,
            semantic_weight=args.semantic_weight,
            bm25_weight=args.bm25_weight
        )
        
        print(f"\nTop {len(results)} results for: '{args.search}'")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']} ({result['source']})")
            print(f"   Score: {result['score']:.3f} (Semantic: {result['semantic_score']:.3f}, BM25: {result['bm25_score']:.3f})")
            print(f"   Doc ID: {result['doc_id']}")
            print(f"   Preview: {result['chunk_text'][:200]}...")
            if result['url']:
                print(f"   URL: {result['url']}")


if __name__ == "__main__":
    main()
