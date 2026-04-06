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
from backend.rag.pipeline import index_document, query_pipeline
from backend.langchain.chat_pipeline import chat
from backend.langchain.memory import reset_memory
from backend.llamaindex.loader import load_documents_from_db
from backend.llamaindex.indexer import build_index
from backend.agents.agent_pipeline import run_agent
logger = get_logger("main")

app = FastAPI(
    title="AI Research Assistant",
    version="0.1.0",
    docs_url=None,  # disable default docs
)

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="AI Research Assistant",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )
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
# Add these imports at the top

# Add these two endpoints at the bottom of main.py

@app.post("/index/{document_id}")
def index_doc(document_id: int, strategy: str = "recursive"):
    """
    Chunks, embeds, and stores a document in ChromaDB.
    Run this once per document before querying.
    
    strategy options: 'fixed', 'recursive', 'semantic'
    """
    logger.info(f"Index request | doc_id={document_id} | strategy={strategy}")
    try:
        result = index_document(document_id, strategy=strategy)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_documents(question: str, document_id: int = None, top_k: int = 3):
    """
    Ask a question. Returns an AI-generated answer with source citations.
    
    document_id: optional — filter to only search within one document
    top_k: how many chunks to use as context (default 3)
    """
    logger.info(f"Query request | question='{question[:50]}' | doc_id={document_id}")
    try:
        result = query_pipeline(question, document_id=document_id, top_k=top_k)
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/chat")
def chat_endpoint(
    question: str,
    session_id: str = "default",
    document_id: int = None,
):
    """
    Conversational endpoint with memory.
    Remembers previous questions in the same session.
    Routes automatically between document search and general knowledge.

    session_id: use same value across turns to maintain conversation.
    """
    logger.info(f"Chat endpoint | session='{session_id}' | question='{question[:50]}'")
    try:
        result = chat(
            question=question,
            session_id=session_id,
            document_id=document_id,
        )
        return result
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/reset")
def reset_chat(session_id: str = "default"):
    """
    Clears conversation memory for a session.
    Call this when the user wants to start a fresh conversation.
    """
    reset_memory(session_id)
    logger.info(f"Memory reset | session='{session_id}'")
    return {"message": f"Conversation memory cleared for session '{session_id}'"}
# add this import at the top with other imports


@app.post("/llamaindex/index")
def llamaindex_index(document_id: int = None):
    """
    Indexes documents into LlamaIndex's ChromaDB collection.
    Different from /index which uses the Stage 3 manual pipeline.
    
    document_id: optional — index one doc, or leave empty to index ALL docs
    """
    logger.info(f"LlamaIndex indexing | doc_id={document_id}")
    try:
        documents = load_documents_from_db(document_id=document_id)
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found in database")
        
        index = build_index(documents)
        return {
            "message": "Documents indexed into LlamaIndex successfully",
            "documents_indexed": len(documents),
            "filenames": [doc.metadata["filename"] for doc in documents],
        }
    except Exception as e:
        logger.error(f"LlamaIndex indexing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/agent/run")
def agent_run(
    question: str,
    session_id: str = "default",
    document_id: int = None,
    show_trace: bool = False,
):
    """
    Autonomous AI agent that decides what tools to use.

    Unlike /query which always does RAG, the agent reasons
    about your question and picks the right approach itself.

    show_trace=true shows you the agent's step-by-step thinking.
    """
    logger.info(f"Agent endpoint | question='{question[:50]}' | session='{session_id}'")
    try:
        result = run_agent(
            question=question,
            session_id=session_id,
            document_id=document_id,
            show_trace=show_trace,
        )
        return result
    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))