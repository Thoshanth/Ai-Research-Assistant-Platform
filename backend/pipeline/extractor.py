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
    text = ""
    page_count = 0
    used_fallback = False

    logger.debug(f"Attempting PyMuPDF extraction | file='{file_path}'")
    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        for i, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            logger.debug(f"Page {i+1}/{page_count} extracted | chars={len(page_text)}")
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e} | will try pdfplumber fallback")

    if len(text.strip()) < 100:
        logger.warning(f"PyMuPDF yielded insufficient text ({len(text.strip())} chars) | switching to pdfplumber")
        used_fallback = True
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            logger.info(f"pdfplumber fallback succeeded | chars={len(text)}")
        except Exception as e:
            logger.error(f"pdfplumber also failed: {e}", exc_info=True)
            raise

    if used_fallback:
        logger.warning("Used pdfplumber fallback — PDF may have complex layout or be scanned")

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