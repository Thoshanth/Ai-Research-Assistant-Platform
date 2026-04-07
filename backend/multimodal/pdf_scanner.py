import fitz  # PyMuPDF
import io
from pathlib import Path
from backend.multimodal.vision_extractor import (
    image_bytes_to_base64,
    extract_text_from_image,
)
from backend.logger import get_logger

logger = get_logger("multimodal.pdf_scanner")

# If a PDF page has fewer than this many characters of text,
# we treat it as a scanned/image page
TEXT_THRESHOLD = 50

# Resolution for rendering PDF pages as images
# 2.0 = 144 DPI (2x default 72 DPI) — good balance of quality vs size
RENDER_SCALE = 2.0


def is_scanned_pdf(pdf_path: str) -> bool:
    """
    Detects if a PDF is scanned (image-based) or digital (text-based).

    Opens the PDF and checks the first 3 pages.
    If they all have very little extractable text,
    the PDF is likely scanned.
    """
    doc = fitz.open(pdf_path)
    pages_to_check = min(3, len(doc))
    scanned_pages = 0

    for i in range(pages_to_check):
        page = doc[i]
        text = page.get_text().strip()
        if len(text) < TEXT_THRESHOLD:
            scanned_pages += 1

    doc.close()

    is_scanned = scanned_pages >= pages_to_check
    logger.info(
        f"PDF scan detection | file='{pdf_path}' | "
        f"pages_checked={pages_to_check} | "
        f"scanned_pages={scanned_pages} | "
        f"is_scanned={is_scanned}"
    )
    return is_scanned


def render_page_to_image(page: fitz.Page) -> bytes:
    """
    Renders a single PDF page as a PNG image in memory.

    Uses a matrix to scale up resolution for better OCR quality.
    Higher scale = better quality but larger file = slower API call.
    RENDER_SCALE=2.0 is a good production balance.
    """
    matrix = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
    pixmap = page.get_pixmap(matrix=matrix)
    return pixmap.tobytes("png")


def extract_text_from_scanned_pdf(pdf_path: str) -> tuple[str, int]:
    """
    Processes a scanned PDF page by page:
    1. Renders each page as a PNG image
    2. Sends to vision LLM for text extraction
    3. Combines all page texts

    Returns (full_text, page_count)
    """
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    logger.info(
        f"Processing scanned PDF | pages={page_count} | file='{pdf_path}'"
    )

    all_text_parts = []

    for page_num in range(page_count):
        logger.info(f"Processing page {page_num + 1}/{page_count}")

        page = doc[page_num]

        # Render page to PNG bytes
        image_bytes = render_page_to_image(page)

        # Convert to base64 for API
        image_base64 = image_bytes_to_base64(image_bytes)

        # Send to vision LLM
        page_text = extract_text_from_image(
            image_base64=image_base64,
            media_type="image/png",
        )

        if page_text.strip():
            all_text_parts.append(
                f"--- Page {page_num + 1} ---\n{page_text}"
            )
            logger.info(
                f"Page {page_num + 1} extracted | chars={len(page_text)}"
            )
        else:
            logger.warning(f"Page {page_num + 1} returned empty text")

    doc.close()

    full_text = "\n\n".join(all_text_parts)
    logger.info(
        f"Scanned PDF complete | total_chars={len(full_text)} | pages={page_count}"
    )
    return full_text, page_count