#!/usr/bin/env python3
"""Quick script to check document count in database"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foia_ai.storage.db import get_session
from foia_ai.storage.models import Document, Page, Source

def main():
    with get_session() as session:
        total_docs = session.query(Document).count()
        total_pages = session.query(Page).count()
        
        sources = session.query(Source).all()
        print(f"Total Documents: {total_docs:,}")
        print(f"Total Pages: {total_pages:,}")
        print("\nBy Source:")
        for source in sources:
            count = session.query(Document).filter_by(source_id=source.id).count()
            print(f"  {source.name}: {count:,} documents")

if __name__ == "__main__":
    main()

