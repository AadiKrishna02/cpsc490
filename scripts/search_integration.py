#!/usr/bin/env python3
"""
Search Integration for FOIA AI Wiki
Integrates hybrid search into the web interface using high-performance Production Search
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.path.insert(0, str(ROOT / "scripts"))
PRODUCTION_SEARCH = False
PRODUCTION_SEARCH_IMPORT_ERROR = None
LazyFederatedSearch = None

try:
    from production_search import get_production_search
    PRODUCTION_SEARCH = True
except ImportError as e:
    PRODUCTION_SEARCH = False
    PRODUCTION_SEARCH_IMPORT_ERROR = str(e)

try:
    from federated_search_lazy import LazyFederatedSearch
except ImportError:
    LazyFederatedSearch = None


class SearchManager:
    """Manages search functionality for the web interface"""
    
    def __init__(self):
        self.search_system = None
        self.index_loaded = False
        self.using_production_search = False
        self.fallback_reason = None
        self.load_system()
    
    def load_system(self):
        """Load search system"""
        try:
            if PRODUCTION_SEARCH:
                print("Initializing Production Search System (LanceDB)...")
                try:
                    self.search_system = get_production_search()
                    has_lancedb = (hasattr(self.search_system, 'table') and 
                                  self.search_system.table is not None)
                    has_tantivy = (hasattr(self.search_system, 'tantivy_searcher') and 
                                  self.search_system.tantivy_searcher is not None)
                    
                    if has_lancedb or has_tantivy:
                        self.index_loaded = True
                        self.using_production_search = True
                        if not has_lancedb:
                            print("Production search: LanceDB missing, using Tantivy only")
                        if not has_tantivy:
                            print("Production search: Tantivy missing, using LanceDB only")
                        
                        if has_lancedb:
                            try:
                                print("Pre-loading embedding model for fast search...")
                                self.search_system._load_model()
                                print("Embedding model pre-loaded")
                            except Exception as model_error:
                                print(f"Failed to pre-load model: {model_error}")
                                print("   Model will load lazily on first search")
                        
                        print("Production search initialized")
                    else:
                        raise Exception("Production search initialized but no usable indexes found (LanceDB and Tantivy both missing)")
                except Exception as prod_error:
                    import traceback
                    print(f"Production search initialization failed: {prod_error}")
                    print(f"   Traceback: {traceback.format_exc()}")
                    print("Falling back to Legacy Federated Search")
                    self.using_production_search = False
                    error_msg = str(prod_error)[:150]  # Limit length for display
                    self.fallback_reason = f"Initialization failed: {error_msg}"
                    try:
                        self._load_legacy()
                    except Exception as legacy_error:
                        print(f"Legacy search also failed: {legacy_error}")
                        self.index_loaded = False
            else:
                print("Production search unavailable, falling back to Legacy Federated Search")
                self.using_production_search = False
                if PRODUCTION_SEARCH_IMPORT_ERROR:
                    error_msg = PRODUCTION_SEARCH_IMPORT_ERROR[:150]
                    self.fallback_reason = f"Production search import failed: {error_msg}"
                else:
                    self.fallback_reason = "Production search module not available (ImportError)"
                self._load_legacy()
                
        except Exception as e:
            print(f"Failed to initialize search: {e}")
            import traceback
            traceback.print_exc()
            self.index_loaded = False
            self.using_production_search = False
            self.fallback_reason = f"Search initialization error: {str(e)[:150]}"

    def _load_legacy(self):
        """Legacy fallback"""
        if LazyFederatedSearch is None:
            print("Legacy federated search not available (ImportError)")
            self.index_loaded = False
            return
            
        batch_dir = ROOT / "data" / "search_indexes"
        if not batch_dir.exists():
            print(f"Legacy search directory not found: {batch_dir}")
            self.index_loaded = False
            return
            
        batch_paths = sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")])
        if not batch_paths:
            print("No legacy indices found")
            self.index_loaded = False
            return
            
        self.search_system = LazyFederatedSearch(batch_paths)
        self.index_loaded = True

    def search(self, query: str, top_k: int = 20, 
               semantic_weight: float = 0.6, diversity: str = "balanced",
               search_mode: str = "hybrid") -> List[Dict]:
        """
        Perform search
        
        Args:
            query: Search query string
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0.0-1.0)
            diversity: Diversity mode ('strict', 'balanced', 'relaxed', 'best')
            search_mode: Search mode ('hybrid', 'semantic', 'bm25')
        """
        if not self.index_loaded or not self.search_system:
            return []
        
        try:
            if PRODUCTION_SEARCH:
                return self.search_system.search(
                    query=query,
                    top_k=top_k,
                    semantic_weight=semantic_weight,
                    diversity=diversity,
                    search_mode=search_mode
                )
            else:
                return self.search_system.search(
                    query=query,
                    top_k=top_k,
                    search_mode="hybrid",  # Legacy doesn't support mode switching
                    semantic_weight=semantic_weight,
                    diversity=diversity
                )
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions based on query"""
        suggestions = []
        if not query or len(query) < 2:
            return suggestions
        
        topics = [
            "CIA operations", "FBI investigations", "NSA surveillance",
            "nuclear weapons", "Cold War", "Soviet Union", "KGB",
            "classified documents", "intelligence gathering",
            "covert operations", "military operations", "DIA reports"
        ]
        
        query_lower = query.lower()
        for topic in topics:
            if query_lower in topic.lower() or topic.lower().startswith(query_lower):
                suggestions.append(topic)
        
        return suggestions[:5]


search_manager = None

def get_search_manager() -> SearchManager:
    """Get the global search manager instance"""
    global search_manager
    
    if search_manager is None:
        search_manager = SearchManager()
    
    return search_manager


def search_documents(query: str, top_k: int = 20) -> List[Dict]:
    """
    Search documents using configured search engine
    """
    manager = get_search_manager()
    
    if not manager.index_loaded:
        return []
    
    results = manager.search(query, top_k)
    
    enhanced_results = []
    for result in results:
        chunk_text = result.get('chunk_text', result.get('text', ''))
        
        enhanced_result = {
            'doc_id': result.get('doc_id', ''),
            'title': result.get('title', 'Untitled'),
            'source': result.get('source', 'Unknown'),
            'url': result.get('url', ''),
            'snippet': chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
            'relevance_score': result.get('score', 0),
            'semantic_score': result.get('semantic_score', 0),
            'bm25_score': result.get('bm25_score', 0),
            'batch_name': result.get('batch_name', 'unknown'),
            'search_type': 'production' if PRODUCTION_SEARCH else 'legacy'
        }
        enhanced_results.append(enhanced_result)
    
    return enhanced_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test search integration")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()
    
    print(f"Testing search integration for: '{args.query}'")
    results = search_documents(args.query, args.top_k)
    
    if results:
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Score: {result['relevance_score']:.3f}")
            print(f"   Source: {result['source']}")
            print(f"   Snippet: {result['snippet']}")
    else:
        print("No results found or search index not available")
