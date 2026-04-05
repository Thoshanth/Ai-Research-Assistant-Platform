import os
import shutil
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from backend.database.db import init_db
from backend.pipeline.extractor import extract_text
from backend.pipeline.cleaner import clean_text, get_word_count
from backend.pipeline.storage import save_document, get_all_documents
from backend.logger import get_logger
# Add this import at the top with the others
from backend.experiments.compare_embeddings import run_comparison

logger = get_logger("main")

app = FastAPI(title="AI Research Assistant", version="0.1.0")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
def startup():
    logger.info("Starting AI Research Assistant API")
    init_db()
    logger.info("Database initialized")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Upload request received | filename='{file.filename}' | content_type='{file.content_type}'")
    start = time.time()

    allowed_types = {"pdf", "csv", "txt"}
    extension = file.filename.split(".")[-1].lower()

    if extension not in allowed_types:
        logger.warning(f"Rejected upload | unsupported type='{extension}' | file='{file.filename}'")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logger.debug(f"File saved to disk | path='{file_path}'")

    file_size_kb = round(os.path.getsize(file_path) / 1024, 2)

    raw_text, page_count = extract_text(str(file_path), extension)
    clean = clean_text(raw_text)
    word_count = get_word_count(clean)

    doc_id = save_document(
        filename=file.filename,
        file_type=extension,
        page_count=page_count,
        word_count=word_count,
        file_size_kb=file_size_kb,
        extracted_text=clean,
    )

    elapsed = round(time.time() - start, 3)
    logger.info(f"Upload pipeline complete | id={doc_id} | file='{file.filename}' | total_time={elapsed}s")

    return {
        "message": "Document processed successfully",
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": extension,
        "page_count": page_count,
        "word_count": word_count,
        "file_size_kb": file_size_kb,
        "processing_time_seconds": elapsed,
    }

@app.get("/documents")
def list_documents():
    logger.info("GET /documents called")
    return get_all_documents()

@app.get("/health")
def health():
    logger.debug("Health check ping")
    return {"status": "ok"}

# Add this new endpoint at the bottom of main.py
@app.post("/experiments/run")
def run_embedding_experiment():
    """
    Triggers the embedding model comparison experiment.
    Results are logged to MLflow and returned as JSON.
    """
    logger.info("Embedding experiment triggered via API")
    try:
        results = run_comparison()
        logger.info(f"Experiment complete | winner='{results.get('winner')}'")
        return {
            "message": "Experiment complete. Run 'mlflow ui' to view dashboard.",
            "results": results
        }
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))