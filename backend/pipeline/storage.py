import time
from datetime import datetime
from backend.database.db import SessionLocal, DocumentRecord
from backend.logger import get_logger

logger = get_logger("pipeline.storage")

def save_document(filename, file_type, page_count, word_count, file_size_kb, extracted_text) -> int:
    logger.info(f"Saving document | file='{filename}' | words={word_count} | size={file_size_kb}KB")
    start = time.time()
    db = SessionLocal()
    try:
        record = DocumentRecord(
            filename=filename,
            file_type=file_type,
            page_count=page_count,
            word_count=word_count,
            file_size_kb=file_size_kb,
            upload_timestamp=datetime.utcnow(),
            extracted_text=extracted_text,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        elapsed = round(time.time() - start, 3)
        logger.info(f"Document saved | id={record.id} | file='{filename}' | time={elapsed}s")
        return record.id
    except Exception as e:
        db.rollback()
        logger.error(f"DB save failed for '{filename}': {e}", exc_info=True)
        raise
    finally:
        db.close()


def get_all_documents() -> list[dict]:
    logger.debug("Fetching all documents from DB")
    db = SessionLocal()
    try:
        records = db.query(DocumentRecord).all()
        logger.info(f"Fetched {len(records)} documents from DB")
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "file_type": r.file_type,
                "page_count": r.page_count,
                "word_count": r.word_count,
                "file_size_kb": r.file_size_kb,
                "upload_timestamp": str(r.upload_timestamp),
            }
            for r in records
        ]
    except Exception as e:
        logger.error(f"DB fetch failed: {e}", exc_info=True)
        raise
    finally:
        db.close()