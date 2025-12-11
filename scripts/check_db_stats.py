
import sys
from pathlib import Path
sys.path.append(str(Path(".").resolve()))
from src.foia_ai.storage.db import get_session
from src.foia_ai.storage.models import Document, Page
from src.foia_ai.retrieval.hybrid_search import HybridRetriever

with get_session() as session:
    doc_count = session.query(Document).count()
    page_count = session.query(Page).count()
    print(f"Total Documents: {doc_count}")
    print(f"Total Pages: {page_count}")

retriever = HybridRetriever()
if retriever.corpus_cache.exists():
    import pickle
    with open(retriever.corpus_cache, 'rb') as f:
        data = pickle.load(f)
        print(f"Cached Pages in Index: {len(data['pages'])}")
else:
    print("No retrieval cache found.")

