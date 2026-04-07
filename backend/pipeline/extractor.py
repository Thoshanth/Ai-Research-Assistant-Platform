import fitz
import pdfplumber
import pandas as pd
import time
from pathlib import Path
from backend.logger import get_logger

logger = get_logger("pipeline.extractor")

def extract_text(file_path: str, file_type: str) -> tuple[str, int | None]:
    logger.info(f"Starting extraction | file='{file_path}' | type='{file_type}'")
    start = time.time()

    if file_type == "pdf":
        result = extract_from_pdf(file_path)
    elif file_type == "csv":
        result = extract_from_csv(file_path)
    elif file_type == "txt":
        result = extract_from_txt(file_path)
    else:
        logger.error(f"Unsupported file type: '{file_type}'")
        raise ValueError(f"Unsupported file type: {file_type}")

    elapsed = round(time.time() - start, 3)
    text, pages = result
    logger.info(f"Extraction complete | file='{Path(file_path).name}' | chars={len(text)} | pages={pages} | time={elapsed}s")
    return result


def extract_from_pdf(file_path: str) -> tuple[str, int]:
    """
    Smart PDF extractor:
    - Digital PDFs → PyMuPDF (fast, Stage 1 approach)
    - Scanned PDFs → Vision LLM (new Stage 8 approach)

    The detection is automatic — users don't need to know
    which type their PDF is.
    """
    from backend.multimodal.pdf_scanner import (
        is_scanned_pdf,
        extract_text_from_scanned_pdf,
    )

    # First check if it's a scanned PDF
    if is_scanned_pdf(file_path):
        logger.info(f"Scanned PDF detected | routing to vision LLM | file='{file_path}'")
        text, page_count = extract_text_from_scanned_pdf(file_path)
        return text, page_count

    # Digital PDF — use original PyMuPDF approach
    logger.info(f"Digital PDF detected | using PyMuPDF | file='{file_path}'")
    text = ""
    page_count = 0

    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        for i, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            logger.debug(f"Page {i+1}/{page_count} | chars={len(page_text)}")
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e} | trying pdfplumber")

    if len(text.strip()) < 100:
        logger.warning("Insufficient text — trying pdfplumber fallback")
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            logger.info("pdfplumber fallback succeeded")
        except Exception as e:
            logger.error(f"pdfplumber also failed: {e}", exc_info=True)
            raise

    return text, page_count


def extract_from_csv(file_path: str) -> tuple[str, None]:
    logger.debug(f"Reading CSV | file='{file_path}'")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"CSV loaded | rows={len(df)} | columns={len(df.columns)} | cols={df.columns.tolist()}")
        text = f"Columns: {', '.join(df.columns.tolist())}\n\n"
        text += f"Total rows: {len(df)}\n\n"
        text += "Sample data (first 50 rows):\n"
        text += df.head(50).to_string(index=False)
        return text, None
    except Exception as e:
        logger.error(f"CSV extraction failed: {e}", exc_info=True)
        raise


def extract_from_txt(file_path: str) -> tuple[str, None]:
    logger.debug(f"Reading TXT | file='{file_path}'")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.debug("TXT read with UTF-8 encoding")
        return text, None
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed | retrying with latin-1")
        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
        logger.info("TXT read successfully with latin-1 fallback encoding")
        return text, None