#!/usr/bin/env python3
"""
Lazy Federated Hybrid Search System

Loads batch indices on-demand to minimize memory usage.
Optimized with parallel batch loading and shared embedding model.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.path.insert(0, str(ROOT / "scripts"))
from hybrid_search_system import HybridSearchSystem
from sentence_transformers import SentenceTransformer


class LazyFederatedSearch:
    """
    Memory-efficient federated search that loads indices on-demand.
    Optimized with:
    - Shared embedding model across all batches
    - Parallel batch loading
    - Batch caching for frequently accessed batches
    """
    
    def __init__(self, batch_paths: List[Path], model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with batch paths (doesn't load indices yet).
        
        Args:
            batch_paths: List of paths to batch index directories
            model_name: SentenceTransformer model name (same for all batches)
        """
        self.batch_paths = batch_paths
        self.batch_info = []
        self.model_name = model_name
        self.shared_embedding_model = None  # Will be loaded on first search
        self.batch_cache = {}  # Cache loaded batches: {batch_name: HybridSearchSystem}
        self.cache_lock = Lock()  # Thread-safe cache access
        self.max_cache_size = 2  # Keep max 2 batches in memory to save RAM
        
        print("Initializing Lazy Federated Search System...")
        print(f"Found {len(batch_paths)} batch indices\n")
        
        for i, batch_path in enumerate(batch_paths, 1):
            try:
                metadata_file = batch_path / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        batch_docs = metadata.get('total_documents', 0)
                        batch_chunks = metadata.get('total_chunks', 0)
                        print(f"{batch_path.name}: {batch_docs:,} docs, {batch_chunks:,} chunks")
                else:
                    print(f"{batch_path.name}: No metadata")
                    batch_docs = 0
                    batch_chunks = 0
                
                self.batch_info.append({
                    'path': batch_path,
                    'name': batch_path.name,
                    'documents': batch_docs,
                    'chunks': batch_chunks
                })
                
            except Exception as e:
                print(f"Error loading metadata for batch {i}: {e}")
        
        total_docs = sum(info['documents'] for info in self.batch_info)
        total_chunks = sum(info['chunks'] for info in self.batch_info)
        
        print(f"\nFederated search ready (lazy loading, parallel enabled)")
        print(f"Total: {len(self.batch_info)} batches, {total_docs:,} documents, {total_chunks:,} chunks")
    
    def _load_shared_embedding_model(self):
        """Load embedding model once, share across all batches"""
        if self.shared_embedding_model is None:
            print(f"Loading shared embedding model: {self.model_name}")
            self.shared_embedding_model = SentenceTransformer(self.model_name)
            print(f"Shared model loaded (will be reused across all batches)")
        return self.shared_embedding_model
    
    def _load_batch_index(self, batch_info: Dict[str, Any]) -> Optional[HybridSearchSystem]:
        """
        Load a single batch index with shared embedding model.
        Uses cache if available.
        """
        batch_name = batch_info['name']
        
        with self.cache_lock:
            if batch_name in self.batch_cache:
                return self.batch_cache[batch_name]
        
        try:
            system = HybridSearchSystem(model_name=self.model_name)
            
            shared_model = self._load_shared_embedding_model()
            
            system.load_index(str(batch_info['path']), shared_embedding_model=shared_model)
            
            with self.cache_lock:
                if len(self.batch_cache) >= self.max_cache_size:
                    oldest = next(iter(self.batch_cache))
                    del self.batch_cache[oldest]
                self.batch_cache[batch_name] = system
            
            return system
        except Exception as e:
            print(f"Error loading batch {batch_name}: {e}")
            return None
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        search_mode: str = "hybrid",
        semantic_weight: float = 0.6,
        diversity: str = "balanced",
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform federated search with parallel batch loading and shared embedding model.
        
        Args:
            query: Search query
            top_k: Number of results to return per batch
            search_mode: "semantic", "bm25", or "hybrid"
            semantic_weight: Weight for semantic scores in hybrid mode (0-1)
            diversity: Diversity mode for results
            parallel: Whether to load and search batches in parallel (default: True)
            
        Returns:
            List of combined and re-ranked results
        """
        if not self.batch_info:
            print("No batch indices found!")
            return []
        
        print(f"\nFederated {search_mode.upper()} Search: '{query}'")
        print(f"Settings: top_k={top_k}, semantic_weight={semantic_weight}, diversity={diversity}")
        print(f"Querying {len(self.batch_info)} batches {'in parallel' if parallel else 'sequentially'}...\n")
        
        shared_model = self._load_shared_embedding_model()
        
        all_results = []
        
        if parallel and len(self.batch_info) > 1:
            max_workers = min(2, len(self.batch_info))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {}
                for batch_info in self.batch_info:
                    future = executor.submit(
                        self._search_single_batch,
                        batch_info,
                        query,
                        top_k,
                        search_mode,
                        semantic_weight,
                        diversity,
                        shared_model
                    )
                    future_to_batch[future] = batch_info
                
                for future in as_completed(future_to_batch):
                    batch_info = future_to_batch[future]
                    try:
                        results = future.result()
                        if results:
                            all_results.extend(results)
                            print(f"{batch_info['name']}: {len(results)} results")
                    except Exception as e:
                        print(f"{batch_info['name']}: Error - {e}")
        else:
            for i, batch_info in enumerate(self.batch_info, 1):
                print(f"[{i}/{len(self.batch_info)}] {batch_info['name']}...")
                try:
                    results = self._search_single_batch(
                        batch_info, query, top_k, search_mode, 
                        semantic_weight, diversity, shared_model
                    )
                    if results:
                        all_results.extend(results)
                        print(f"Found {len(results)} results")
                except Exception as e:
                    print(f"Error: {e}")
        
        final_results = self._rerank_results(all_results, top_k, diversity)
        
        print(f"\nRetrieved {len(final_results)} combined results")
        
        gc.collect()
        
        return final_results
    
    def _search_single_batch(
        self,
        batch_info: Dict[str, Any],
        query: str,
        top_k: int,
        search_mode: str,
        semantic_weight: float,
        diversity: str,
        shared_model: Any
    ) -> List[Dict[str, Any]]:
        """
        Search a single batch. Internal helper for parallel execution.
        
        Returns:
            List of search results with batch info added
        """
        system = self._load_batch_index(batch_info)
        if system is None:
            return []
        
        if search_mode == "hybrid":
            bm25_weight = 1.0 - semantic_weight
            results = system.hybrid_search(
                query,
                top_k=top_k,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                diversity_mode=diversity
            )
        elif search_mode == "semantic":
            raw_results = system.search_semantic(query, top_k=top_k)
            results = system._convert_indices_to_results(raw_results)
        else:  # bm25
            raw_results = system.search_bm25(query, top_k=top_k)
            results = system._convert_indices_to_results(raw_results)
        
        for result in results:
            result['batch_name'] = batch_info['name']
            result['batch_path'] = str(batch_info['path'])
        
        return results
    
    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        diversity: str = "balanced"
    ) -> List[Dict[str, Any]]:
        """
        Re-rank and deduplicate combined results with diversity filtering.
        """
        if not results:
            return []
        
        scores = [r.get('score', 0) for r in results]
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for result in results:
                original_score = result.get('score', 0)
                result['normalized_score'] = (original_score - min_score) / score_range
        
        seen = {}
        for result in results:
            doc_id = result.get('doc_id', '')
            chunk_text = result.get('chunk_text', result.get('text', ''))[:100]  # First 100 chars as fingerprint
            key = (doc_id, chunk_text)
            
            if key not in seen or result.get('normalized_score', 0) > seen[key].get('normalized_score', 0):
                seen[key] = result
        
        deduplicated = list(seen.values())
        
        deduplicated.sort(key=lambda x: x.get('normalized_score', 0), reverse=True)
        
        diversity_limits = {
            'strict': 1,      # Max 1 chunk per document
            'balanced': 2,    # Max 2 chunks per document (default)
            'relaxed': 3,     # Max 3 chunks per document
            'best': 999       # No limit
        }
        max_chunks_per_doc = diversity_limits.get(diversity, 2)
        
        final_results = []
        doc_chunk_count = {}
        
        for result in deduplicated:
            if len(final_results) >= top_k:
                break
            
            doc_id = result.get('doc_id', '')
            
            if doc_chunk_count.get(doc_id, 0) >= max_chunks_per_doc:
                continue
            
            doc_chunk_count[doc_id] = doc_chunk_count.get(doc_id, 0) + 1
            final_results.append(result)
        
        unique_docs = len(set(r.get('doc_id', '') for r in final_results))
        avg_chunks_per_doc = len(final_results) / unique_docs if unique_docs > 0 else 0
        print(f"Final results: {len(final_results)} chunks from {unique_docs} unique documents (avg: {avg_chunks_per_doc:.1f} chunks/doc, diversity: {diversity})")
        
        return final_results


def load_federated_system() -> Optional[LazyFederatedSearch]:
    """Load batch index paths (lazy loading)."""
    batch_dir = ROOT / "data" / "search_indexes"
    
    batch_paths = sorted([
        p for p in batch_dir.iterdir()
        if p.is_dir() and p.name.startswith("batch_")
    ])
    
    if not batch_paths:
        print("No batch indices found!")
        print(f"   Searched in: {batch_dir}")
        return None
    
    return LazyFederatedSearch(batch_paths)


def main():
    """Interactive federated search CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lazy Federated Hybrid Search")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results")
    parser.add_argument("--mode", "-m", choices=["semantic", "bm25", "hybrid"], 
                       default="hybrid", help="Search mode")
    parser.add_argument("--semantic-weight", "-w", type=float, default=0.6,
                       help="Semantic weight for hybrid search (0-1)")
    parser.add_argument("--diversity", "-d", 
                       choices=["strict", "balanced", "relaxed", "best"],
                       default="balanced", help="Diversity mode")
    
    args = parser.parse_args()
    
    system = load_federated_system()
    if not system:
        return
    
    if not args.query:
        print("\n" + "="*80)
        print("Lazy Federated Search - Interactive Mode")
        print("="*80)
        print("\nCommands:")
        print("  /mode [semantic|bm25|hybrid]  - Change search mode")
        print("  /weight [0-1]                  - Set semantic weight")
        print("  /diversity [mode]              - Set diversity mode")
        print("  /k [number]                    - Set number of results")
        print("  /quit or /exit                 - Exit")
        print("\nOr just type your search query!\n")
        
        search_mode = args.mode
        top_k = args.top_k
        semantic_weight = args.semantic_weight
        diversity = args.diversity
        
        while True:
            try:
                query = input("\nSearch> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['/quit', '/exit']:
                    print("Goodbye!")
                    break
                
                if query.startswith('/'):
                    parts = query.split()
                    cmd = parts[0].lower()
                    
                    if cmd == '/mode' and len(parts) > 1:
                        if parts[1] in ['semantic', 'bm25', 'hybrid']:
                            search_mode = parts[1]
                            print(f"Mode set to: {search_mode}")
                        else:
                            print("Invalid mode. Use: semantic, bm25, or hybrid")
                    
                    elif cmd == '/weight' and len(parts) > 1:
                        try:
                            w = float(parts[1])
                            if 0 <= w <= 1:
                                semantic_weight = w
                                print(f"Semantic weight set to: {semantic_weight}")
                            else:
                                print("Weight must be between 0 and 1")
                        except ValueError:
                            print("Invalid weight value")
                    
                    elif cmd == '/diversity' and len(parts) > 1:
                        if parts[1] in ['strict', 'balanced', 'relaxed', 'best']:
                            diversity = parts[1]
                            print(f"Diversity set to: {diversity}")
                        else:
                            print("Invalid diversity mode")
                    
                    elif cmd == '/k' and len(parts) > 1:
                        try:
                            top_k = int(parts[1])
                            print(f"Top-k set to: {top_k}")
                        except ValueError:
                            print("Invalid number")
                    
                    else:
                        print("Unknown command")
                    
                    continue
                
                results = system.search(
                    query,
                    top_k=top_k,
                    search_mode=search_mode,
                    semantic_weight=semantic_weight,
                    diversity=diversity
                )
                
                print("\n" + "="*80)
                print(f"Top {len(results)} Results")
                print("="*80)
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. [{result.get('batch_name', 'unknown')}] Score: {result.get('score', 0):.4f} (norm: {result.get('normalized_score', 0):.4f})")
                    print(f"   Document: {result.get('title', 'Untitled')}")
                    print(f"   Source: {result.get('source', 'Unknown')}")
                    print(f"   URL: {result.get('url', 'N/A')}")
                    text = result.get('chunk_text', result.get('text', ''))
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"   Text: {preview}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    else:
        results = system.search(
            args.query,
            top_k=args.top_k,
            search_mode=args.mode,
            semantic_weight=args.semantic_weight,
            diversity=args.diversity
        )
        
        print("\n" + "="*80)
        print(f"Top {len(results)} Results")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result.get('batch_name', 'unknown')}] Score: {result.get('score', 0):.4f} (norm: {result.get('normalized_score', 0):.4f})")
            print(f"   Document: {result.get('title', 'Untitled')}")
            print(f"   Source: {result.get('source', 'Unknown')}")
            print(f"   URL: {result.get('url', 'N/A')}")
            text = result.get('chunk_text', result.get('text', ''))
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"   Text: {preview}")


if __name__ == "__main__":
    main()

