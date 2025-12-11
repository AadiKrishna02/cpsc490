import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/foia_ai.db")
BLOB_DIR = Path(os.getenv("BLOB_DIR", "data/blob"))
USER_AGENT = os.getenv("USER_AGENT", "FOIA-AI-Bot/0.1")
RATE_LIMIT_PER_SEC = float(os.getenv("RATE_LIMIT_PER_SEC", "0.5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
RETRY_TOTAL = int(os.getenv("RETRY_TOTAL", "3"))

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
OCR_LANG = os.getenv("OCR_LANG", "eng")

ENABLE_CIA_CREST = os.getenv("ENABLE_CIA_CREST", "true").lower() == "true"
ENABLE_FBI_VAULT = os.getenv("ENABLE_FBI_VAULT", "true").lower() == "true"
ENABLE_DIA_RR = os.getenv("ENABLE_DIA_RR", "true").lower() == "true"

BLOB_DIR.mkdir(parents=True, exist_ok=True)
