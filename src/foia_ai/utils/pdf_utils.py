from __future__ import annotations

from pathlib import Path
from typing import Optional

from pdfminer.high_level import extract_text


def extract_text_len(pdf_path: Path, maxpages: Optional[int] = None) -> int:
    """Return length of extractable text using pdfminer.six. If parsing fails, return 0."""
    try:
        text = extract_text(str(pdf_path), maxpages=maxpages)
        return len(text.strip()) if text else 0
    except Exception:
        return 0


def is_scanned_like(pdf_path: Path, *, threshold: int = 100) -> bool:
    """
    Heuristic: consider PDF scanned-like if extractable text length < threshold.
    threshold can be tuned; 100 chars is a conservative default.
    """
    return extract_text_len(pdf_path) < threshold
