#!/usr/bin/env python3
"""
Build search index in batches and merge them
This allows processing large document collections without memory issues
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pickle
import numpy as np
import faiss
import gc

sys.path.insert(0, 'scripts')
from hybrid_search_system import HybridSearchSystem

sys.path.insert(0, 'src')
from foia_ai.storage.db import get_session
from foia_ai.storage.models import Document


def get_total_document_count():
    """Get total number of documents in database"""
    with get_session() as session:
        return session.query(Document).count()


def build_batch_index(batch_num, offset, limit, embed_batch_size, chunk_process_size):
    """Build index for a single batch of documents"""
    print("\n" + "="*80)
    print(f"Building Index for Batch {batch_num}")
    print(f"   Documents: {offset} to {offset + limit - 1}")
    print("="*80)
    
    search_system = HybridSearchSystem()
    
    print(f"Loading documents {offset} to {offset + limit}...")
    
    from foia_ai.storage.models import Page
    
    with get_session() as session:
        documents_query = session.query(Document).offset(offset).limit(limit).all()
        
        print(f"Processing {len(documents_query):,} documents...")
        
        documents = []
        for doc in documents_query:
            pages = session.query(Page).filter_by(document_id=doc.id).order_by(Page.page_no).all()
            pages_with_text = [p for p in pages if p.text]
            
            if not pages_with_text:
                continue
            
            page_texts = []
            for page in pages_with_text:
                page_texts.append({
                    'page_no': page.page_no,
                    'text': page.text
                })
            
            full_text = "\n\n".join(p['text'] for p in page_texts)
            if len(full_text.strip()) < 100:
                continue
            
            documents.append({
                'id': doc.external_id,
                'title': doc.title or f"Document {doc.external_id}",
                'source': doc.source.name if doc.source else 'Unknown',
                'text': full_text,
                'url': doc.url,
                'page_count': len(pages_with_text),
                'word_count': len(full_text.split()),
                'pages': page_texts  # Store page-level data for citation tracking
            })
    
    if not documents:
        print(f"No valid documents found in batch {batch_num}")
        return None
    
    print(f"Loaded {len(documents):,} documents with text")
    
    start_time = datetime.now()
    search_system.build_search_index(
        documents,
        chunk_size=512,
        encode_batch_size=embed_batch_size,
        chunk_processing_size=chunk_process_size,
    )
    duration = datetime.now() - start_time
    
    batch_index_name = f"batch_{batch_num:03d}_index"
    batch_path = search_system.save_index(batch_index_name)
    
    print(f"Batch {batch_num} completed in {duration}")
    print(f"   Documents: {len(documents):,}")
    print(f"   Chunks: {len(search_system.document_chunks):,}")
    print(f"   Saved to: {batch_path}")
    
    return {
        'batch_num': batch_num,
        'offset': offset,
        'limit': limit,
        'doc_count': len(documents),
        'chunk_count': len(search_system.document_chunks),
        'duration': str(duration),
        'path': batch_path
    }


def merge_batch_indices(batch_stats, output_name="full_search_index"):
    """Merge all batch indices into a single unified index"""
    print("\n" + "="*80)
    print("Merging All Batch Indices")
    print("="*80)
    
    if not batch_stats:
        print("No batch indices to merge")
        return
    
    print(f"Merging {len(batch_stats)} batch indices...")
    
    all_document_chunks = []
    all_chunk_metadata = []
    all_faiss_vectors = []
    
    for i, stats in enumerate(batch_stats, 1):
        batch_path = Path(stats['path'])
        print(f"\nLoading batch {stats['batch_num']} ({i}/{len(batch_stats)})...")
        print(f"   Path: {batch_path}")
        
        try:
            chunks_file = batch_path / "document_chunks.pkl"
            if chunks_file.exists():
                with open(chunks_file, 'rb') as f:
                    chunks = pickle.load(f)
                    all_document_chunks.extend(chunks)
            
            metadata_file = batch_path / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'chunk_metadata' in metadata:
                        all_chunk_metadata.extend(metadata['chunk_metadata'])
            
            # Note: We'll tokenize from chunks later (BM25 doesn't expose corpus)
            
            faiss_file = batch_path / "semantic.faiss"
            if faiss_file.exists():
                batch_index = faiss.read_index(str(faiss_file))
                batch_vectors = np.zeros((batch_index.ntotal, batch_index.d), dtype='float32')
                for j in range(batch_index.ntotal):
                    batch_vectors[j] = batch_index.reconstruct(j)
                all_faiss_vectors.append(batch_vectors)
                print(f"Loaded {batch_index.ntotal:,} vectors")
                del batch_index
                del batch_vectors
                gc.collect()
            
        except Exception as e:
            print(f"Error loading batch {stats['batch_num']}: {e}")
            continue
    
    print(f"\nMerge Statistics:")
    print(f"   Total chunks: {len(all_document_chunks):,}")
    print(f"   Total metadata entries: {len(all_chunk_metadata):,}")
    print(f"   FAISS vector batches: {len(all_faiss_vectors)}")
    
    print("\nBuilding merged FAISS index...")
    if all_faiss_vectors:
        merged_vectors = np.vstack(all_faiss_vectors)
        print(f"   Total vectors: {merged_vectors.shape[0]:,}")
        
        dimension = merged_vectors.shape[1]
        merged_faiss = faiss.IndexFlatIP(dimension)
        
        block_size = 10000
        for start in range(0, len(merged_vectors), block_size):
            end = min(start + block_size, len(merged_vectors))
            block = merged_vectors[start:end]
            merged_faiss.add(block)
            print(f"   Added {end:,}/{len(merged_vectors):,} vectors...")
        
        del merged_vectors
        del all_faiss_vectors
        gc.collect()
        
        print(f"Merged FAISS index: {merged_faiss.ntotal:,} vectors")
    else:
        print("No FAISS vectors to merge")
        return
    
    print("\nBuilding merged BM25 index...")
    print(f"   Tokenizing {len(all_document_chunks):,} chunks...")
    from rank_bm25 import BM25Okapi
    
    tokenized_chunks = []
    batch_size = 100000
    for i in range(0, len(all_document_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_document_chunks))
        batch = all_document_chunks[i:end_idx]
        tokenized_batch = [chunk.lower().split() for chunk in batch]
        tokenized_chunks.extend(tokenized_batch)
        print(f"   Tokenized {end_idx:,}/{len(all_document_chunks):,} chunks ({end_idx/len(all_document_chunks)*100:.1f}%)")
        del tokenized_batch
        gc.collect()
    
    print("   Creating BM25 index...")
    merged_bm25 = BM25Okapi(tokenized_chunks)
    del tokenized_chunks
    gc.collect()
    print(f"Merged BM25 index: {len(all_document_chunks):,} chunks")
    
    print("\nSaving merged index...")
    output_path = Path("data/search_indexes") / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    faiss_path = output_path / "semantic.faiss"
    faiss.write_index(merged_faiss, str(faiss_path))
    print(f"Saved FAISS index: {merged_faiss.ntotal:,} vectors")
    
    bm25_path = output_path / "bm25.pkl"
    with open(bm25_path, 'wb') as f:
        pickle.dump(merged_bm25, f)
    print(f"Saved BM25 index")
    
    chunks_path = output_path / "document_chunks.pkl"
    with open(chunks_path, 'wb') as f:
        pickle.dump(all_document_chunks, f)
    print(f"Saved {len(all_document_chunks):,} chunks")
    
    import json
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_documents': sum(s['doc_count'] for s in batch_stats),
        'total_chunks': len(all_document_chunks),
        'embedding_dimension': dimension,
        'chunk_size': 512,
        'overlap': 50,
        'batches_merged': len(batch_stats),
        'chunk_metadata': all_chunk_metadata,
        'semantic_weight': 0.6,
        'bm25_weight': 0.4
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata")
    
    print("\n" + "="*80)
    print("Index Merge Complete!")
    print("="*80)
    print(f"Final Statistics:")
    print(f"   Total Documents: {metadata['total_documents']:,}")
    print(f"   Total Chunks: {metadata['total_chunks']:,}")
    print(f"   FAISS Vectors: {merged_faiss.ntotal:,}")
    print(f"   Saved to: {output_path}")
    print("="*80)
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Build search index in batches")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Number of documents per batch (default: 1000)")
    parser.add_argument("--embed-batch-size", type=int, default=8,
                       help="SentenceTransformer encode batch size (default: 8)")
    parser.add_argument("--chunk-process-size", type=int, default=128,
                       help="Number of chunks to process before adding to FAISS (default: 128)")
    parser.add_argument("--start-batch", type=int, default=1,
                       help="Starting batch number (for resuming)")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Maximum number of batches to process (for testing)")
    parser.add_argument("--merge-only", action="store_true",
                       help="Only merge existing batch indices, don't build new ones")
    args = parser.parse_args()
    
    print("="*80)
    print("Batched Search Index Builder")
    print("="*80)
    
    total_docs = get_total_document_count()
    total_batches = (total_docs + args.batch_size - 1) // args.batch_size
    
    print(f"Index Build Plan:")
    print(f"   Total Documents: {total_docs:,}")
    print(f"   Batch Size: {args.batch_size:,} documents")
    print(f"   Total Batches: {total_batches}")
    print(f"   Starting Batch: {args.start_batch}")
    if args.max_batches:
        print(f"   Max Batches: {args.max_batches}")
    print()
    
    batch_stats = []
    
    if not args.merge_only:
        batch_num = args.start_batch
        offset = (batch_num - 1) * args.batch_size
        
        while offset < total_docs:
            if args.max_batches and batch_num >= args.start_batch + args.max_batches:
                print(f"\nReached max batches limit ({args.max_batches})")
                break
            
            remaining = total_docs - offset
            batch_limit = min(args.batch_size, remaining)
            
            try:
                stats = build_batch_index(
                    batch_num,
                    offset,
                    batch_limit,
                    args.embed_batch_size,
                    args.chunk_process_size
                )
                
                if stats:
                    batch_stats.append(stats)
                    
                    stats_file = Path("data/search_indexes/batch_build_progress.pkl")
                    stats_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(stats_file, 'wb') as f:
                        pickle.dump(batch_stats, f)
                
            except Exception as e:
                print(f"\nError building batch {batch_num}: {e}")
                print("   Saving progress and continuing...")
                import traceback
                traceback.print_exc()
            
            batch_num += 1
            offset += args.batch_size
            
            gc.collect()
    
    else:
        print("Loading existing batch indices...")
        stats_file = Path("data/search_indexes/batch_build_progress.pkl")
        if stats_file.exists():
            with open(stats_file, 'rb') as f:
                batch_stats = pickle.load(f)
            print(f"Found {len(batch_stats)} existing batch indices")
        else:
            print("No batch progress file found. Run without --merge-only first.")
            return
    
    if batch_stats:
        print(f"\nBuilt {len(batch_stats)} batches successfully")
        merge_batch_indices(batch_stats, output_name="full_search_index")
    else:
        print("\nNo batches were built successfully")


if __name__ == "__main__":
    main()

