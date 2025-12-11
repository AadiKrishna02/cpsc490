"""Text cleanup utilities to improve extracted text quality."""

import re
import unicodedata


def clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted text from PDFs.
    
    Handles:
    - Unicode normalization
    - Extra whitespace
    - Common OCR errors
    - Encoding issues
    - Line break artifacts
    """
    if not text:
        return ""
    
    text = unicodedata.normalize('NFKD', text)
    
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    replacements = {
        '\x00': '',
        '\ufffd': '',
        '\u200b': '',
        '\u200c': '',
        '\u200d': '',
        '\ufeff': '',
        '': '',
        '': '',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    text = re.sub(r' +', ' ', text)
    
    text = re.sub(r'\n\n+', '\n\n', text)
    
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    ocr_fixes = {
        r'\bl\b': 'I',
        r'\b0\b': 'O',
        r'rn': 'm',
        r'\|': 'l',
    }
    
    for pattern, replacement in ocr_fixes.items():
        text = re.sub(pattern, replacement, text)
    
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'-{4,}', '---', text)
    text = re.sub(r'_{4,}', '___', text)
    
    return text.strip()


def remove_headers_footers(text: str) -> str:
    """
    Attempt to remove repeated headers and footers from multi-page text.
    
    This is a heuristic approach - looks for lines that appear repeatedly
    at similar positions across pages.
    """
    lines = text.split('\n')
    
    if len(lines) < 10:
        return text
    
    line_counts = {}
    for line in lines:
        clean_line = line.strip()
        if len(clean_line) < 50 and clean_line:
            line_counts[clean_line] = line_counts.get(clean_line, 0) + 1
    
    repeated_lines = {line for line, count in line_counts.items() if count >= 3 and len(line) < 30}
    
    filtered_lines = [line for line in lines if line.strip() not in repeated_lines]
    
    return '\n'.join(filtered_lines)


def fix_common_pdf_artifacts(text: str) -> str:
    """
    Fix common PDF extraction artifacts.
    
    - Spurious single characters on their own lines
    - Broken bullet points
    - Inconsistent spacing around punctuation
    """
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= 3 or len(stripped) == 0:
            filtered.append(line)
    
    text = '\n'.join(filtered)
    
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation if missing
    
    text = re.sub(r'^\s*[•·∙◦▪▫–-]\s*', '• ', text, flags=re.MULTILINE)
    
    return text


def enhance_text_quality(text: str) -> str:
    """
    Main entry point: apply all text quality enhancements.
    """
    text = clean_extracted_text(text)
    text = remove_headers_footers(text)
    text = fix_common_pdf_artifacts(text)
    return text
