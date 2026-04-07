import os
import time
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from prometheus_fastapi_instrumentator import Instrumentator
from backend.database.db import init_db
from backend.pipeline.extractor import extract_text
from backend.pipeline.cleaner import clean_text, get_word_count
from backend.pipeline.storage import save_document, get_all_documents
from backend.experiments.compare_embeddings import run_comparison
from backend.multimodal.image_handler import process_image_file, SUPPORTED_IMAGE_TYPES
from backend.rag.pipeline import index_document, query_pipeline
from backend.langchain.chat_pipeline import chat
from backend.langchain.memory import reset_memory
from backend.guardrails.pipeline import run_input_guards, run_output_guards
from backend.agents.agent_pipeline import run_agent
from backend.llamaindex.loader import load_documents_from_db
from backend.llamaindex.indexer import build_index
from backend.middleware.monitoring import log_request, get_system_stats
from backend.logger import get_logger
from backend.graphrag.pipeline import (
    build_knowledge_graph,
    query_knowledge_graph,
    explore_graph,
)

logger = get_logger("main")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("Starting AI Research Assistant API")
    logger.info("=" * 50)
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down AI Research Assistant API")


# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="AI Research Assistant",
    version="0.1.0",
    docs_url=None,
    lifespan=lifespan,
)

# ── Prometheus metrics (auto-instruments all endpoints) ───────────
Instrumentator().instrument(app).expose(app)


# ── Request timing middleware ─────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """
    Runs around every single HTTP request.
    Measures how long each request takes and logs it.
    This is how you find slow endpoints.
    """
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    log_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    # Add timing header so clients can see latency too
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response


# ── Custom Swagger UI (fixes blank page issue) ────────────────────
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="AI Research Assistant",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


# ══════════════════════════════════════════════════════════════════
# STAGE 1 — Document Upload & Processing
# ══════════════════════════════════════════════════════════════════

@app.post("/upload", tags=["Stage 1 - Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, CSV, or TXT file. Extracts and stores text."""
    logger.info(f"Upload | file='{file.filename}'")
    allowed = {"pdf", "csv", "txt"}
    extension = file.filename.split(".")[-1].lower()

    if extension not in allowed:
        raise HTTPException(400, f"Unsupported file type: {extension}")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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

    return {
        "message": "Document processed successfully",
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": extension,
        "page_count": page_count,
        "word_count": word_count,
        "file_size_kb": file_size_kb,
    }


@app.get("/documents", tags=["Stage 1 - Ingestion"])
def list_documents():
    """List all processed documents."""
    return get_all_documents()


# ══════════════════════════════════════════════════════════════════
# STAGE 2 — Embedding Experiments
# ══════════════════════════════════════════════════════════════════

@app.post("/experiments/run", tags=["Stage 2 - Experiments"])
def run_embedding_experiment():
    """Compare embedding models and log results to MLflow."""
    logger.info("Embedding experiment triggered")
    try:
        results = run_comparison()
        return {
            "message": "Experiment complete. Run 'mlflow ui' to view dashboard.",
            "results": results,
        }
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════
# STAGE 3 — RAG Pipeline
# ══════════════════════════════════════════════════════════════════

@app.post("/index/{document_id}", tags=["Stage 3 - RAG"])
def index_doc(document_id: int, strategy: str = "recursive"):
    """Chunk, embed and store a document in ChromaDB for RAG."""
    logger.info(f"Index request | doc_id={document_id} | strategy={strategy}")
    try:
        return index_document(document_id, strategy=strategy)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/query", tags=["Stage 3 - RAG"])
def query_documents(question: str, document_id: int = None, top_k: int = 3):
    """Ask a question. Returns AI answer with source citations."""
    logger.info(f"Query | question='{question[:50]}'")

    # ← INPUT GUARDS
    run_input_guards(question)

    try:
        result = query_pipeline(question, document_id=document_id, top_k=top_k)

        # ← OUTPUT GUARDS (expect citations for RAG)
        safe = run_output_guards(result["answer"], expect_citations=True)
        result["answer"] = safe["answer"]
        result["guardrail_info"] = {
            "warnings": safe["guardrail_warnings"],
            "pii_redacted": safe["pii_redacted"],
            "hallucination_warning": safe["hallucination_warning"],
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════
# STAGE 4 — LlamaIndex + Conversation Memory
# ══════════════════════════════════════════════════════════════════

@app.post("/llamaindex/index", tags=["Stage 4 - Frameworks"])
def llamaindex_index(document_id: int = None):
    """Index documents into LlamaIndex's ChromaDB collection."""
    logger.info(f"LlamaIndex indexing | doc_id={document_id}")
    try:
        documents = load_documents_from_db(document_id=document_id)
        if not documents:
            raise HTTPException(404, "No documents found in database")
        index = build_index(documents)
        return {
            "message": "Indexed into LlamaIndex successfully",
            "documents_indexed": len(documents),
            "filenames": [d.metadata["filename"] for d in documents],
        }
    except Exception as e:
        logger.error(f"LlamaIndex indexing failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/chat", tags=["Stage 4 - Frameworks"])
def chat_endpoint(
    question: str,
    session_id: str = "default",
    document_id: int = None,
):
    """Conversational endpoint with memory and automatic routing."""
    logger.info(f"Chat | session='{session_id}' | question='{question[:50]}'")

    # ← INPUT GUARDS
    run_input_guards(question)

    try:
        result = chat(
            question=question,
            session_id=session_id,
            document_id=document_id,
        )

        # ← OUTPUT GUARDS
        safe = run_output_guards(result["answer"])
        result["answer"] = safe["answer"]
        result["guardrail_info"] = {
            "warnings": safe["guardrail_warnings"],
            "pii_redacted": safe["pii_redacted"],
            "hallucination_warning": safe["hallucination_warning"],
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))

@app.post("/chat/reset", tags=["Stage 4 - Frameworks"])
def reset_chat(session_id: str = "default"):
    """Clear conversation memory for a session."""
    reset_memory(session_id)
    return {"message": f"Memory cleared for session '{session_id}'"}


# ══════════════════════════════════════════════════════════════════
# STAGE 5 — AI Agent
# ══════════════════════════════════════════════════════════════════
@app.post("/agent/run", tags=["Stage 5 - Agents"])
def agent_run(
    question: str,
    session_id: str = "default",
    document_id: int = None,
    show_trace: bool = False,
):
    """Autonomous agent that reasons and picks tools itself."""
    logger.info(f"Agent | question='{question[:50]}' | session='{session_id}'")

    # ← INPUT GUARDS
    run_input_guards(question)

    try:
        result = run_agent(
            question=question,
            session_id=session_id,
            document_id=document_id,
            show_trace=show_trace,
        )

        # ← OUTPUT GUARDS
        safe = run_output_guards(result["answer"])
        result["answer"] = safe["answer"]
        result["guardrail_info"] = {
            "warnings": safe["guardrail_warnings"],
            "pii_redacted": safe["pii_redacted"],
            "hallucination_warning": safe["hallucination_warning"],
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))

# ══════════════════════════════════════════════════════════════════
# STAGE 6 — Health + Monitoring
# ══════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Stage 6 - Monitoring"])
def health():
    """
    Enhanced health check.
    Returns system stats so you can monitor resource usage.
    Monitoring tools ping this every 30 seconds.
    """
    stats = get_system_stats()
    db_ok = True
    try:
        docs = get_all_documents()
        doc_count = len(docs)
    except Exception:
        db_ok = False
        doc_count = 0

    status = "healthy" if db_ok else "degraded"
    logger.debug(f"Health check | status={status} | cpu={stats['cpu_percent']}%")

    return {
        "status": status,
        "version": "0.1.0",
        "database": "ok" if db_ok else "error",
        "documents_stored": doc_count,
        "system": stats,
    }


@app.get("/metrics/summary", tags=["Stage 6 - Monitoring"])
def metrics_summary():
    """
    Returns a human-readable summary of app metrics.
    For detailed Prometheus metrics go to GET /metrics
    """
    stats = get_system_stats()
    docs = get_all_documents()
    return {
        "total_documents": len(docs),
        "system_resources": stats,
        "endpoints": {
            "prometheus_metrics": "/metrics",
            "health_check": "/health",
            "api_docs": "/docs",
        },
    }


@app.post("/upload/image", tags=["Stage 8 - Multimodal"])
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file (JPG, PNG, WEBP).

    The vision LLM will:
    1. Extract any text visible in the image
    2. Generate a full description of the image content

    Both are stored and become searchable via /query.
    Perfect for: photos of documents, screenshots,
    charts, diagrams, handwritten notes.
    """
    logger.info(f"Image upload | file='{file.filename}'")

    extension = file.filename.split(".")[-1].lower()
    if extension not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            400,
            f"Unsupported image type: {extension}. "
            f"Supported: {list(SUPPORTED_IMAGE_TYPES.keys())}"
        )

    # Save image to uploads folder
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"Image saved | path='{file_path}'")

    # Process with vision LLM
    result = process_image_file(str(file_path))

    # Store in SQLite so it's queryable via RAG
    doc_id = save_document(
        filename=file.filename,
        file_type=extension,
        page_count=1,
        word_count=len(result["extracted_text"].split()),
        file_size_kb=result["file_size_kb"],
        extracted_text=result["extracted_text"],
    )

    logger.info(f"Image document stored | doc_id={doc_id}")

    return {
        "message": "Image processed successfully",
        "document_id": doc_id,
        "filename": file.filename,
        "file_type": extension,
        "description_preview": result["description"][:200] + "...",
        "text_extracted_chars": len(result["text_only"]),
        "word_count": len(result["extracted_text"].split()),
        "file_size_kb": result["file_size_kb"],
        "next_step": f"Index this document: POST /index/{doc_id}",
    }

# ══════════════════════════════════════════════════════════════════
# STAGE 9 — GraphRAG & Knowledge Graphs
# ══════════════════════════════════════════════════════════════════

@app.post("/graphrag/build/{document_id}", tags=["Stage 9 - GraphRAG"])
def build_graph_endpoint(document_id: int):
    """
    Extracts entities and relationships from a document
    and builds a knowledge graph.

    Run this after /index/{document_id}.
    Takes 1-2 minutes depending on document size.
    """
    logger.info(f"GraphRAG build | doc_id={document_id}")
    try:
        result = build_knowledge_graph(document_id)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Graph build failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/graphrag/query", tags=["Stage 9 - GraphRAG"])
def graphrag_query_endpoint(
    question: str,
    document_id: int,
    top_k: int = 3,
):
    """
    Query using combined graph traversal + vector search.

    Better than /query for relationship questions like:
    - "How is X related to Y?"
    - "What connects A and B?"
    - "Which entities are most connected?"
    """
    logger.info(f"GraphRAG query | question='{question[:50]}'")

    # Input guards still apply
    run_input_guards(question)

    try:
        result = query_knowledge_graph(question, document_id, top_k)

        # Output guards still apply
        safe = run_output_guards(result["answer"])
        result["answer"] = safe["answer"]
        result["guardrail_info"] = {
            "warnings": safe["guardrail_warnings"],
            "pii_redacted": safe["pii_redacted"],
        }
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"GraphRAG query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/graphrag/explore", tags=["Stage 9 - GraphRAG"])
def explore_graph_endpoint(document_id: int):
    """
    Explore the knowledge graph for a document.
    Returns all entities grouped by type and all relationships.
    Great for understanding what the graph extracted.
    """
    logger.info(f"Graph explore | doc_id={document_id}")
    try:
        return explore_graph(document_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Graph explore failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))