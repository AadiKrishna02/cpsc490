#!/usr/bin/env python3
"""Interactive database query tool for FOIA AI corpus."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foia_ai.storage.db import get_session
from foia_ai.storage.models import Source, Document, Page


def show_database_overview():
    """Show overview of the database contents."""
    print("FOIA AI Database Overview")
    print("=" * 50)
    
    with get_session() as session:
        sources = session.query(Source).all()
        documents = session.query(Document).all()
        pages = session.query(Page).all()
        
        print(f"Sources: {len(sources)}")
        for source in sources:
            doc_count = session.query(Document).filter_by(source_id=source.id).count()
            print(f"  • {source.name}: {doc_count} documents")
        
        print(f"\nDocuments: {len(documents)}")
        print(f"Pages with text: {len(pages)}")
        
        total_words = 0
        for page in pages:
            if page.text:
                total_words += len(page.text.split())
        
        print(f"Total words: {total_words:,}")


def search_text(query: str, limit: int = 10):
    """Search for text across all pages."""
    print(f"\nSearching for: '{query}'")
    print("-" * 40)
    
    with get_session() as session:
        pages = session.query(Page).filter(
            Page.text.ilike(f'%{query}%')
        ).limit(limit).all()
        
        if not pages:
            print("No results found.")
            return
        
        for page in pages:
            doc = page.document
            source = doc.source
            
            text = page.text or ""
            query_lower = query.lower()
            text_lower = text.lower()
            
            if query_lower in text_lower:
                start = text_lower.find(query_lower)
                context_start = max(0, start - 100)
                context_end = min(len(text), start + len(query) + 100)
                context = text[context_start:context_end]
                
                print(f"{doc.external_id} (Page {page.page_no})")
                print(f"   Source: {source.name}")
                print(f"   Context: ...{context}...")
                print()


def show_document_details(external_id: str):
    """Show detailed information about a specific document."""
    with get_session() as session:
        doc = session.query(Document).filter_by(external_id=external_id).first()
        
        if not doc:
            print(f"Document '{external_id}' not found.")
            return
        
        pages = session.query(Page).filter_by(document_id=doc.id).order_by(Page.page_no).all()
        
        print(f"\nDocument: {doc.external_id}")
        print(f"Title: {doc.title}")
        print(f"Source: {doc.source.name}")
        print(f"URL: {doc.url}")
        print(f"File: {doc.file_path}")
        print(f"Pages: {len(pages)}")
        
        total_words = sum(len(p.text.split()) if p.text else 0 for p in pages)
        print(f"Total words: {total_words:,}")
        
        print(f"\nPages:")
        for page in pages[:5]:  # Show first 5 pages
            method = "OCR" if page.ocr_confidence else "Text"
            word_count = len(page.text.split()) if page.text else 0
            preview = (page.text or "")[:150] + "..." if page.text and len(page.text) > 150 else (page.text or "")
            
            print(f"  Page {page.page_no} ({method}): {word_count} words")
            print(f"    {preview}")
            print()


def list_documents():
    """List all documents in the database."""
    print("\nAll Documents")
    print("-" * 30)
    
    with get_session() as session:
        documents = session.query(Document).all()
        
        for doc in documents:
            page_count = session.query(Page).filter_by(document_id=doc.id).count()
            total_words = 0
            
            if page_count > 0:
                pages = session.query(Page).filter_by(document_id=doc.id).all()
                total_words = sum(len(p.text.split()) if p.text else 0 for p in pages)
            
            print(f"{doc.external_id}")
            print(f"   Title: {doc.title}")
            print(f"   Source: {doc.source.name}")
            print(f"   Pages: {page_count}, Words: {total_words:,}")
            print()


def interactive_mode():
    """Interactive query mode."""
    print("\nInteractive Query Mode")
    print("Commands:")
    print("  search <query>     - Search text across all documents")
    print("  doc <external_id>  - Show details for a specific document")
    print("  list              - List all documents")
    print("  overview          - Show database overview")
    print("  quit              - Exit")
    print()
    
    while True:
        try:
            command = input("foia-ai> ").strip()
            
            if command == "quit":
                break
            elif command == "overview":
                show_database_overview()
            elif command == "list":
                list_documents()
            elif command.startswith("search "):
                query = command[7:]
                search_text(query)
            elif command.startswith("doc "):
                external_id = command[4:]
                show_document_details(external_id)
            elif command == "":
                continue
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "overview":
            show_database_overview()
        elif command == "list":
            list_documents()
        elif command == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            search_text(query)
        elif command == "doc" and len(sys.argv) > 2:
            external_id = sys.argv[2]
            show_document_details(external_id)
        else:
            print("Usage: python query_database.py [overview|list|search <query>|doc <id>]")
    else:
        show_database_overview()
        interactive_mode()


if __name__ == "__main__":
    main()
