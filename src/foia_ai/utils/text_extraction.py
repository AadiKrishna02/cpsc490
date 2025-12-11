from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO

from .text_cleanup import enhance_text_quality

LOGGER = logging.getLogger(__name__)


def extract_text_by_page(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Extract text from PDF page by page for text-based PDFs.
    Returns list of (page_number, text) tuples.
    """
    try:
        pages_text = []
        
        with open(pdf_path, 'rb') as file:
            resource_manager = PDFResourceManager()
            laparams = LAParams(
                word_margin=0.1,
                char_margin=2.0,
                line_margin=0.5,
                boxes_flow=0.5
            )
            
            for page_num, page in enumerate(PDFPage.get_pages(file), 1):
                output_string = StringIO()
                converter = TextConverter(resource_manager, output_string, laparams=laparams)
                interpreter = PDFPageInterpreter(resource_manager, converter)
                
                interpreter.process_page(page)
                text = output_string.getvalue()
                
                converter.close()
                output_string.close()
                
                text = enhance_text_quality(text)
                if text:
                    pages_text.append((page_num, text))
        
        return pages_text
        
    except Exception as e:
        LOGGER.error("Failed to extract text by page from %s: %s", pdf_path, e)
        return []


def extract_text_with_metadata(pdf_path: Path) -> dict:
    """
    Extract text with additional metadata useful for analysis
    Returns dict with text, word_count, char_count, page_count, and avg_words_per_page
    """
    try:
        full_text = extract_text(str(pdf_path))
        
        pages = extract_text_by_page(pdf_path)
        
        word_count = len(full_text.split()) if full_text else 0
        char_count = len(full_text) if full_text else 0
        page_count = len(pages)
        
        return {
            'full_text': full_text,
            'pages': pages,
            'word_count': word_count,
            'char_count': char_count,
            'page_count': page_count,
            'avg_words_per_page': word_count / page_count if page_count > 0 else 0
        }
        
    except Exception as e:
        LOGGER.error("Failed to extract text with metadata from %s: %s", pdf_path, e)
        return {
            'full_text': '',
            'pages': [],
            'word_count': 0,
            'char_count': 0,
            'page_count': 0,
            'avg_words_per_page': 0
        }


def chunk_text_for_retrieval(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    Useful for very long pages or documents.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            sentence_end = text.rfind('.', end - 100, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks
