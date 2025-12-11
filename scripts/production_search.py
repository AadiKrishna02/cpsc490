#!/usr/bin/env python3
"""
Production Hybrid Search System
Powered by LanceDB (Vector) and Tantivy (Keyword)
Replaces the old FAISS/Pickle system for 100x speed and scalability.
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import lancedb
    import tantivy
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing dependencies. Please run: pip install lancedb tantivy sentence-transformers")

def setup_tantivy_schema():
    """Define the schema for Tantivy keyword search index (must match migration script)"""
    schema_builder = tantivy.SchemaBuilder()
    
    schema_builder.add_text_field("chunk_text", stored=True)  # The actual content
    schema_builder.add_text_field("title", stored=True)       # Document title
    schema_builder.add_text_field("doc_id", stored=True)      # ID for filtering
    schema_builder.add_unsigned_field("chunk_idx", stored=True) # To link back to original
    schema_builder.add_text_field("batch_name", stored=True)  # Source batch
    
    return schema_builder.build()

class ProductionSearchSystem:
    """
    High-performance hybrid search system using LanceDB and Tantivy.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
        
        self.data_dir = ROOT / "data"
        self.lancedb_path = self.data_dir / "lancedb_store"
        self.tantivy_path = self.data_dir / "tantivy_store"
        
        self.db = None
        self.table = None
        self.tantivy_index = None
        self.tantivy_searcher = None
        
        self._initialize_stores()
        
    def _initialize_stores(self):
        """Initialize connections to LanceDB and Tantivy"""
        try:
            if self.lancedb_path.exists():
                self.db = lancedb.connect(self.lancedb_path)
                if "chunks" in self.db.table_names():
                    self.table = self.db.open_table("chunks")
                    print(f"Lancedb connected ({self.table.count_rows()} rows)")
            
            if self.tantivy_path.exists():
                try:
                    schema = setup_tantivy_schema()
                    self.tantivy_index = tantivy.Index(
                        schema,
                        path=str(self.tantivy_path)
                    )
                    self.tantivy_index.reload()
                    self.tantivy_searcher = self.tantivy_index.searcher()
                    print(f"Tantivy connected ({self.tantivy_searcher.num_docs} docs)")
                except Exception as tantivy_error:
                    print(f"Tantivy schema mismatch or corruption: {tantivy_error}")
                    print(f"   Deleting corrupted index at {self.tantivy_path} - will rebuild on next migration")
                    import shutil
                    if self.tantivy_path.exists():
                        shutil.rmtree(self.tantivy_path)
                    self.tantivy_path.mkdir(parents=True, exist_ok=True)
                    self.tantivy_index = None
                    self.tantivy_searcher = None
                
        except Exception as e:
            print(f"Error initializing search stores: {e}")

    def _load_model(self):
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
    
    def _get_document_metadata(self, doc_id: str) -> Dict[str, str]:
        """Look up document metadata from database by doc_id (external_id)"""
        try:
            from foia_ai.storage.db import get_session
            from foia_ai.storage.models import Document
            
            with get_session() as session:
                doc = session.query(Document).filter_by(external_id=doc_id).first()
                if doc:
                    source_name = doc.source.name if doc.source else 'Unknown'
                    url = f"/pdf/{doc_id}"
                    return {
                        'title': doc.title or f"{doc_id}.pdf",
                        'source': source_name,
                        'url': url
                    }
        except Exception as e:
            print(f"Error looking up document metadata for {doc_id}: {e}")
        
        return {
            'title': f"{doc_id}.pdf",
            'source': 'Unknown',
            'url': f"/pdf/{doc_id}"
        }

    def search(self, query: str, top_k: int = 20, 
               semantic_weight: float = 0.6, diversity: str = "balanced",
               search_mode: str = "hybrid") -> List[Dict]:
        """
        Perform high-speed search using LanceDB + Tantivy
        """
        results = []
        semantic_hits = []
        bm25_results = []
        
        if search_mode in ["semantic", "hybrid"] and self.table:
            self._load_model()
            query_vec = self.embedding_model.encode(query).tolist()
            
            semantic_hits = self.table.search(query_vec) \
                .metric("cosine") \
                .limit(max(top_k * 10, 100)) \
                .to_list()
            
            
            if semantic_hits:
                raw_scores = []
                for hit in semantic_hits:
                    distance = hit.get('_distance', 0)
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                    raw_scores.append(similarity)
                    hit['semantic_score_raw'] = similarity
                    hit['chunk_text'] = hit.get('text', hit.get('chunk_text', ''))  # Standardize key
                    
                    doc_id = hit.get('doc_id', '')
                    if doc_id and (not hit.get('url') or not hit.get('source')):
                        doc_metadata = self._get_document_metadata(doc_id)
                        if not hit.get('url'):
                            hit['url'] = doc_metadata['url']
                        if not hit.get('source'):
                            hit['source'] = doc_metadata['source']
                        if not hit.get('title'):
                            hit['title'] = doc_metadata['title']
                    elif doc_id:
                        hit['url'] = f"/pdf/{doc_id}"
                
                min_score = min(raw_scores)
                max_score = max(raw_scores)
                score_range = max_score - min_score if max_score > min_score else 1.0
                
                for i, hit in enumerate(semantic_hits):
                    raw = raw_scores[i]
                    normalized = (raw - min_score) / score_range if score_range > 0 else 1.0
                    hit['semantic_score'] = normalized
                    hit['score'] = normalized  # Default if only semantic
                
                if search_mode == "semantic":
                    results = semantic_hits
            else:
                results = []

        if search_mode in ["bm25", "hybrid"] and self.tantivy_searcher:
            try:
                query_parser = self.tantivy_index.parse_query(query, ["chunk_text", "title"])
                search_result = self.tantivy_searcher.search(query_parser, max(top_k * 10, 100))
                
                for score, doc_address in search_result.hits:
                    doc = self.tantivy_searcher.doc(doc_address)
                    
                    def get_field(doc, field_name, default=''):
                        try:
                            values = doc[field_name]
                            return values[0] if values else default
                        except (KeyError, IndexError, TypeError):
                            return default
                    
                    doc_id = get_field(doc, 'doc_id')
                    doc_metadata = self._get_document_metadata(doc_id)
                    
                    res = {
                        'chunk_text': get_field(doc, 'chunk_text'),
                        'title': doc_metadata['title'],  # Use database title if available
                        'doc_id': doc_id,
                        'bm25_score': score,
                        'score': score, # Default if only bm25
                        'url': doc_metadata['url'],  # Construct URL consistently
                        'source': doc_metadata['source']  # Look up source from database
                    }
                    bm25_results.append(res)
                
                print(f"BM25 found {len(bm25_results)} results")
                    
                if search_mode == "bm25":
                    if bm25_results:
                        raw_bm25_scores = [hit.get('bm25_score', 0) for hit in bm25_results]
                        min_score = min(raw_bm25_scores)
                        max_score = max(raw_bm25_scores)
                        score_range = max_score - min_score if max_score > min_score else 1.0
                        
                        for i, hit in enumerate(bm25_results):
                            raw = raw_bm25_scores[i]
                            normalized = (raw - min_score) / score_range if score_range > 0 else 1.0
                            normalized = max(0.0, min(1.0, normalized))
                            hit['bm25_score_raw'] = raw
                            hit['bm25_score'] = normalized
                            hit['score'] = normalized
                            hit['semantic_score'] = 0
                    results = bm25_results
                    
            except Exception as e:
                print(f"Tantivy search error: {e}")
                import traceback
                traceback.print_exc()

        if search_mode == "hybrid":
            print(f"Hybrid search: semantic={len(semantic_hits)} results, bm25={len(bm25_results)} results")
            bm25_weight = 1.0 - semantic_weight
            
            if semantic_hits and bm25_results:
                def normalize_raw_scores(score_list):
                    """Normalize a list of raw scores to [0, 1] range using min-max"""
                    if not score_list:
                        return {}
                    scores = [s for s in score_list if s is not None]
                    if not scores:
                        return {}
                    min_s = min(scores)
                    max_s = max(scores)
                    if max_s == min_s:
                        return {i: 1.0 for i in range(len(score_list))}
                    return {i: (s - min_s) / (max_s - min_s) 
                           for i, s in enumerate(score_list)}
                
                raw_semantic_scores = [hit.get('semantic_score_raw', hit.get('semantic_score', 0)) 
                                      for hit in semantic_hits]
                raw_bm25_scores = [hit.get('bm25_score', 0) for hit in bm25_results]
                
                sem_norm_map = normalize_raw_scores(raw_semantic_scores)
                bm25_norm_map = normalize_raw_scores(raw_bm25_scores)
                
                def create_result_key(result):
                    """Create a key for matching exact same chunks"""
                    doc_id = result.get('doc_id', '')
                    chunk_text = result.get('chunk_text', result.get('text', ''))
                    normalized_text = ' '.join(chunk_text.strip().lower().split())
                    return (doc_id, normalized_text)
                
                semantic_index = {}
                for i, hit in enumerate(semantic_hits):
                    key = create_result_key(hit)
                    semantic_index[key] = {
                        'index': i,
                        'result': hit,
                        'norm_score': sem_norm_map.get(i, 0)
                    }
                
                bm25_index = {}
                for i, hit in enumerate(bm25_results):
                    key = create_result_key(hit)
                    bm25_index[key] = {
                        'index': i,
                        'result': hit,
                        'norm_score': bm25_norm_map.get(i, 0),
                        'raw_score': raw_bm25_scores[i]
                    }
                
                
                combined_map = {}
                all_keys = set(semantic_index.keys()) | set(bm25_index.keys())
                
                def normalize_bm25_for_lookup(raw_score):
                    if not raw_bm25_scores: return 0
                    min_s, max_s = min(raw_bm25_scores), max(raw_bm25_scores)
                    if max_s == min_s:
                        return 1.0 if raw_score == max_s else 0
                    normalized = (raw_score - min_s) / (max_s - min_s)
                    return max(0.0, min(1.0, normalized))

                def normalize_sem_for_lookup(raw_score):
                    if not raw_semantic_scores: return 0
                    min_s, max_s = min(raw_semantic_scores), max(raw_semantic_scores)
                    if max_s == min_s:
                        return 1.0 if raw_score == max_s else 0
                    normalized = (raw_score - min_s) / (max_s - min_s)
                    return max(0.0, min(1.0, normalized))

                lookups_performed = 0
                MAX_LOOKUPS = 50  # Safety limit
                
                for key in all_keys:
                    sem_data = semantic_index.get(key)
                    bm25_data = bm25_index.get(key)
                    
                    base_result = sem_data['result'] if sem_data else bm25_data['result']
                    doc_id = base_result.get('doc_id', '')
                    chunk_text = base_result.get('chunk_text', base_result.get('text', ''))
                    
                    known_sem_score = sem_data['norm_score'] if sem_data else None
                    known_bm25_score = bm25_data['norm_score'] if bm25_data else None
                    known_bm25_raw = bm25_data['raw_score'] if bm25_data else None
                    
                    
                    should_lookup_bm25 = (
                        known_bm25_score is None and 
                        sem_data and sem_data['index'] < 25 and 
                        lookups_performed < MAX_LOOKUPS
                    )
                    
                    should_lookup_sem = (
                        known_sem_score is None and 
                        bm25_data and bm25_data['index'] < 25 and 
                        lookups_performed < MAX_LOOKUPS
                    )
                    
                    if should_lookup_bm25:
                        try:
                            safe_text = chunk_text[:200].replace('"', '\\"').replace(':', '\\:')
                            chunk_query = f'doc_id:"{doc_id}" AND chunk_text:"{safe_text}"'
                            lookup_parser = self.tantivy_index.parse_query(chunk_query, ["doc_id", "chunk_text"])
                            lookup_result = self.tantivy_searcher.search(lookup_parser, 1)
                            
                            if lookup_result.hits:
                                score, _ = list(lookup_result.hits)[0]
                                known_bm25_raw = score
                                known_bm25_score = normalize_bm25_for_lookup(score)
                                lookups_performed += 1
                        except Exception:
                            pass

                    if should_lookup_sem:
                        try:
                            chunk_vec = self.embedding_model.encode([chunk_text])[0].tolist()
                            chunk_result = self.table.search(chunk_vec).metric("cosine").limit(1).to_list()
                            
                            if chunk_result:
                                hit = chunk_result[0]
                                if hit.get('doc_id') == doc_id:
                                    distance = hit.get('_distance', 2.0)
                                    raw_sim = max(0.0, 1.0 - (distance / 2.0))
                                    known_sem_score = normalize_sem_for_lookup(raw_sim)
                                    lookups_performed += 1
                        except Exception:
                            pass

                    final_sem_score = known_sem_score if known_sem_score is not None else 0
                    final_bm25_score = known_bm25_score if known_bm25_score is not None else 0
                    final_bm25_raw = known_bm25_raw if known_bm25_raw is not None else 0
                    
                    combined_score = (semantic_weight * final_sem_score + 
                                    bm25_weight * final_bm25_score)
                    
                    combined_map[key] = {
                        **base_result,
                        'semantic_score': final_sem_score,
                        'bm25_score': final_bm25_score,
                        'bm25_score_raw': final_bm25_raw,
                        'combined_score': combined_score
                    }
                
                results = sorted(combined_map.values(), 
                               key=lambda x: x.get('combined_score', 0), 
                               reverse=True)
                
                for r in results:
                    r['score'] = r.get('combined_score', 0)
                    
            elif semantic_hits:
                print("BM25 failed/missing, using semantic-only results")
                raw_semantic_scores = [hit.get('semantic_score_raw', hit.get('semantic_score', 0)) 
                                      for hit in semantic_hits]
                sem_norm_map = normalize_raw_scores(raw_semantic_scores)
                for i, hit in enumerate(semantic_hits):
                    hit['semantic_score'] = sem_norm_map.get(i, 0)
                    hit['score'] = sem_norm_map.get(i, 0)
                    hit['bm25_score'] = 0
                    hit['bm25_score_raw'] = 0
                results = semantic_hits
            elif bm25_results:
                print("Semantic failed/missing, using BM25-only results")
                raw_bm25_scores = [hit.get('bm25_score', 0) for hit in bm25_results]
                bm25_norm_map = normalize_raw_scores(raw_bm25_scores)
                for i, hit in enumerate(bm25_results):
                    hit['bm25_score_raw'] = raw_bm25_scores[i]
                    hit['bm25_score'] = bm25_norm_map.get(i, 0)
                    hit['semantic_score'] = 0
                    hit['score'] = bm25_norm_map.get(i, 0)
                results = bm25_results
            else:
                results = []

        final_results = self._apply_diversity(results, top_k, diversity)
        return final_results

    def _apply_diversity(self, results, top_k, diversity):
        """Apply diversity filtering to results"""
        if not results: return []
        
        diversity_limits = {
            'strict': 1,
            'balanced': 2,
            'relaxed': 3,
            'best': 999
        }
        max_chunks = diversity_limits.get(diversity, 2)
        
        final = []
        doc_counts = defaultdict(int)
        
        for res in results:
            if len(final) >= top_k: break
            
            doc_id = res.get('doc_id', 'unknown')
            if doc_counts[doc_id] >= max_chunks:
                continue
                
            doc_counts[doc_id] += 1
            final.append(res)
            
        return final

_system = None

def get_production_search():
    global _system
    if _system is None:
        _system = ProductionSearchSystem()
    return _system

if __name__ == "__main__":
    s = get_production_search()
    res = s.search("CIA operations in Cuba", top_k=5)
    for r in res:
        print(f"- {r['title']}: {r['score']:.3f}")

