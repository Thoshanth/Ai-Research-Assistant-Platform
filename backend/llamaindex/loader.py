from llama_index.core import Document
from backend.database.db import SessionLocal, DocumentRecord
from backend.logger import get_logger

logger = get_logger("llamaindex.loader")


def load_documents_from_db(document_id: int = None) -> list[Document]:
    """
    Loads extracted text from SQLite (populated by Stage 1)
    and converts them into LlamaIndex Document objects.

    LlamaIndex Document is just a text + metadata wrapper.
    It knows nothing about files — it just holds text and
    lets LlamaIndex handle chunking and indexing from there.
    """
    db = SessionLocal()
    try:
        query = db.query(DocumentRecord)
        if document_id:
            query = query.filter(DocumentRecord.id == document_id)

        records = query.all()

        if not records:
            logger.warning(f"No documents found | doc_id filter={document_id}")
            return []

        documents = []
        for record in records:
            doc = Document(
                text=record.extracted_text,
                metadata={
                    "document_id": record.id,
                    "filename": record.filename,
                    "file_type": record.file_type,
                    "word_count": record.word_count,
                    "upload_timestamp": str(record.upload_timestamp),
                },
                doc_id=f"doc_{record.id}",
            )
            documents.append(doc)
            logger.info(f"Loaded document | id={record.id} | file='{record.filename}' | words={record.word_count}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    finally:
        db.close()