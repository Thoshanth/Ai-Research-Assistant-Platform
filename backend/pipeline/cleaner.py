import re
import time
from backend.logger import get_logger

logger = get_logger("pipeline.cleaner")

def clean_text(raw_text: str) -> str:
    logger.debug(f"Starting text cleaning | input_chars={len(raw_text)}")
    start = time.time()

    text = raw_text
    text = text.replace("\x00", "")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()

    elapsed = round(time.time() - start, 3)
    chars_removed = len(raw_text) - len(text)

    logger.info(f"Cleaning complete | input_chars={len(raw_text)} | output_chars={len(text)} | removed={chars_removed} | time={elapsed}s")

    if len(text.strip()) == 0:
        logger.warning("Cleaning resulted in empty text — file may be scanned/image-only")

    return text


def get_word_count(text: str) -> int:
    count = len(text.split())
    logger.debug(f"Word count: {count}")
    return count
