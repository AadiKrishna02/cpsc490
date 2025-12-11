from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from pdf2image import convert_from_path
import pytesseract

from ..config import TESSERACT_CMD, OCR_LANG, BLOB_DIR
from ..utils.text_cleanup import enhance_text_quality

LOGGER = logging.getLogger(__name__)


pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy.
    - Convert to grayscale
    - Increase contrast
    - Apply thresholding
    """
    import cv2
    import numpy as np
    
    img_array = np.array(img)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return Image.fromarray(binary)


def ocr_pdf_to_pages(pdf_path: Path, *, out_dir: Path | None = None, dpi: int = 300) -> List[Tuple[int, str, float]]:
    """
    Convert a PDF to images and run Tesseract OCR per page with preprocessing.

    Args:
        pdf_path: Path to PDF file
        out_dir: Output directory for intermediate images
        dpi: DPI for PDF to image conversion (higher = better quality, default 300)

    Returns list of (page_no, text, confidence_estimate)
    """
    pdf_path = Path(pdf_path)
    if out_dir is None:
        out_dir = (BLOB_DIR / "ocr" / pdf_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(str(pdf_path), dpi=dpi)
    results: List[Tuple[int, str, float]] = []
    
    custom_config = r'--oem 3 --psm 1'  # OEM 3 = LSTM, PSM 1 = automatic page segmentation with OSD
    
    for idx, img in enumerate(images, start=1):
        img_path = out_dir / f"page_{idx:04d}.png"
        
        try:
            preprocessed = preprocess_image_for_ocr(img)
            preprocessed.save(img_path)
        except Exception as e:
            LOGGER.warning("Preprocessing failed for page %d, using original: %s", idx, e)
            img.save(img_path)
        
        text = pytesseract.image_to_string(
            Image.open(img_path), 
            lang=OCR_LANG,
            config=custom_config
        )
        
        text = enhance_text_quality(text)
        
        conf = 0.0
        try:
            data = pytesseract.image_to_data(
                Image.open(img_path), 
                lang=OCR_LANG, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            confs = [float(c) for c in data.get("conf", []) if c not in ("-1", "-1.0")]
            if confs:
                conf = sum(confs) / len(confs)
        except Exception as e:
            LOGGER.warning("Failed to compute confidence for %s: %s", img_path, e)
        
        results.append((idx, text, conf))
    
    return results
