# Civis Intelligence - Quick Start Guide

A Retrieval-Augmented Generation (RAG) system for analyzing FOIA and generating context-aware wiki pages.

## Quick Start (5 minutes)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/foia-ai.git
cd foia-ai
pip install -r requirements.txt
```

### 2. Download Pre-built Search Indexes

Our search system requires pre-built indexes (~8GB download):

```bash
./scripts/download_indexes.sh
```

This downloads:
- **LanceDB** (4.2GB) - Vector search with embeddings
- **Tantivy** (4.1GB) - Keyword search (BM25)

**Note**: Download takes 10-30 minutes depending on your connection.

### 3. Run the Website

```bash
python scripts/wiki_web.py
```

Open **http://127.0.0.1:5053** in your browser.

---

## Features

- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search
- **Context-Aware Wiki Generation**: Generate comprehensive wiki pages from FOIA documents
- **Citation Validation**: Automatically validates and scores citations
- **Iterative Refinement**: Improves content quality through multiple refinement passes
- **PDF Analysis**: Upload and analyze your own declassified documents

---

## What's Included

### Generated Wiki Pages

The `data/wiki/` directory contains pre-generated wiki pages on topics like:
- CIA Operations in Cuba
- DIA Reports on Military Activities
- Counterinsurgency Operations
- Intelligence Community Activities
- And many more...

### Source Data

- **6,000+ declassified documents** from:
  - DIA Reading Room
  - Government Attic

---

## Architecture

### Search System (Production)

```
Query → Hybrid Search
         ├── LanceDB (Vector/Semantic Search)
         │   └── all-MiniLM-L6-v2 embeddings
         └── Tantivy (Keyword/BM25 Search)
              └── Full-text index

Combined → Ranked Results → Context Retrieval
```

### Wiki Generation Pipeline

```
Topic → Search → Retrieve Context → LLM Generation
                                      ↓
                               Citation Validation
                                      ↓
                               Quality Check
                                      ↓
                           Pass? → Save : Refine → Retry
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `wiki_web.py` | Main Flask web application |
| `production_search.py` | High-performance search system (LanceDB + Tantivy) |
| `context_aware_wiki_generator.py` | Generate wiki pages with citations |
| `download_indexes.sh` | Download pre-built search indexes |
| `build_index_batched.py` | Build search indexes from PDFs |

---

## System Requirements

### Minimum
- **RAM**: 8GB
- **Disk**: 25GB free (11GB for indexes, 14GB for processing)
- **Python**: 3.9+

### Recommended
- **RAM**: 16GB+ (for faster embedding generation)
- **Disk**: 50GB+ (if building indexes yourself)
- **CPU**: Multi-core (index building is CPU-intensive)

---

## Cloud Deployment

The indexes are hosted on **Cloudflare R2**:

If deploying to production:
1. Use environment variables for API keys
2. Consider using PostgreSQL instead of SQLite
3. Run behind a production WSGI server (gunicorn, uwsgi)

---

## Troubleshooting

### Download Fails

```bash
cd data
wget -c https://pub-9a2bbfbefc804ee6b876c870df20d034.r2.dev/indexes/lancedb_store.tar.gz
```

### Out of Disk Space

You need ~25GB free. Clean up:
```bash
# Remove compressed files after extraction
rm data/*.tar.gz
```

### Search Not Working

```bash
# Check if indexes exist
ls -lh data/lancedb_store data/tantivy_store

# If missing, re-download
./scripts/download_indexes.sh
```

### Website Won't Start

```bash
# Check if port 5053 is in use
lsof -i :5053

# Kill existing process
kill <PID>

# Restart
python scripts/wiki_web.py
```

---

## Documentation

- **[SETUP_WITH_INDEXES.md](SETUP_WITH_INDEXES.md)** - Detailed setup guide

---

## Contributing

Contributions welcome! Areas of interest:
- Additional document sources
- Improved citation extraction
- Better search ranking
- UI/UX improvements


---

## Acknowledgments

Built with:
- **LanceDB** - Vector database
- **Tantivy** - Full-text search engine  
- **Sentence Transformers** - Embeddings
- **OpenAI** - LLM for generation
- **Flask** - Web framework

