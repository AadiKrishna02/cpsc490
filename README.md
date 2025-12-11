# FOIA AI: Retrieval-First Pipeline for FOIA Corpora

This repository contains a modular pipeline to ingest FOIA repositories (GovernmentAttic.org and DIA FOIA Reading Room), OCR scanned PDFs, normalize metadata, and persist structured data for retrieval and wiki-style topic pages.

## Features (v0)
- Document ingestion from GovernmentAttic.org and DIA FOIA Reading Room
- Support for additional sources (CIA CREST, FBI Vault code available but not currently in use)
- Config via environment variables (`.env`)
- Storage layer with SQLAlchemy models for sources, documents, and pages
- Local dev with SQLite; production-ready for Postgres via `DATABASE_URL`
- OCR pipeline (Tesseract) with optional preprocessing hooks
- Scripts for DB bootstrap and pilot ingestion

## Quickstart

1) Create and activate a Python 3.10+ environment

2) Install dependencies
```
pip install -r requirements.txt
```

3) Copy the environment template and set values
```
cp .env.example .env
```

By default, the app uses a local SQLite DB at `data/foia_ai.db` and stores files under `data/blob/`.
To use Postgres, set `DATABASE_URL` accordingly, e.g.:
```
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/foia_ai
```

4) Initialize the database
```
python scripts/bootstrap_db.py
```

5) Run a small pilot ingestion (example)
```
python scripts/pilot_ingest.py --source dia-reading-room --limit 10
```

## Project Structure
```
.
├── README.md
├── requirements.txt
├── .env.example
├── Makefile
├── data/
│   ├── blob/               # blob storage for PDFs/images
│   └── foia_ai.db          # local SQLite (dev)
├── scripts/
│   ├── bootstrap_db.py
│   └── pilot_ingest.py
└── src/
    └── foia_ai/
        ├── __init__.py
        ├── config.py
        ├── logging_setup.py
        ├── storage/
        │   ├── __init__.py
        │   ├── db.py
        │   └── models.py
        ├── ingest/
        │   ├── __init__.py
        │   ├── common.py
        │   ├── dia_reading_room.py
        │   ├── process_governmentattic.py
        │   ├── cia_crest.py (available, not currently in use)
        │   └── fbi_vault.py (available, not currently in use)
        └── ocr/
            ├── __init__.py
            └── pipeline.py
```

## Notes
- Respect `robots.txt` and terms of use for each source. Configure rate limits via env vars.
- Tesseract must be installed on your system for OCR (e.g., `brew install tesseract` on macOS).
- For Postgres, ensure `psycopg2-binary` works in your environment or install `psycopg2` from source.

## Roadmap
- Implement full crawlers with resume, dedupe, and robust error handling
- Add evaluation harness for OCR quality (WER) and retrieval metrics
- Hybrid retrieval (BM25 + dense) and dispersion measurement
- Summarization with citation enforcement and attribution checking
- Wiki-style topic page builder
