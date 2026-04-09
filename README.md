markdown# 🔬 AI Research Assistant Platform

A production-grade AI engineering project built stage by stage — from raw document ingestion to multi-agent collaboration. This platform demonstrates every core skill required for an AI Engineer role in 2025.

---

## 🎯 What This Project Does

Upload any document (PDF, CSV, TXT, or even scanned images) and ask questions about it in natural language. The system retrieves relevant information, generates cited answers, remembers your conversation, and can deploy autonomous agents to perform deep multi-step research.

Built as a **learning progression** — each stage adds a new production AI engineering concept on top of the previous one.

---

## 🏗️ Architecture Overview
User Request
│
▼
Stage 7: Safety Guardrails (input validation, PII, topic, output)
│
▼
FastAPI Backend (10 integrated stages)
│
├── Stage 1:  Document Ingestion (PDF/CSV/TXT extraction)
├── Stage 2:  Embedding Experiments (MLflow tracking)
├── Stage 3:  RAG Pipeline (ChromaDB + hybrid search)
├── Stage 4:  LlamaIndex + Conversation Memory
├── Stage 5:  ReAct Agent (tool-calling)
├── Stage 6:  Docker + Prometheus monitoring
├── Stage 7:  Guardrails & Safety
├── Stage 8:  Multimodal (vision LLM for images/scanned PDFs)
├── Stage 9:  GraphRAG (knowledge graph + two-path retrieval)
└── Stage 10: Multi-Agent System (LangGraph)

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.11 |
| **LLM** | Groq (Llama 3.1 8B Instant) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB (persistent) |
| **RAG Framework** | LlamaIndex |
| **Agent Framework** | LangChain + LangGraph |
| **Knowledge Graph** | NetworkX |
| **Experiment Tracking** | MLflow |
| **Vision/Multimodal** | Groq Vision (Llama 4 Scout) |
| **Database** | SQLite + SQLAlchemy |
| **Monitoring** | Prometheus + psutil |
| **Deployment** | Docker + Docker Compose |

---

## 🚀 Stages Built

### Stage 1 — Data Ingestion Pipeline
- Extracts text from PDF, CSV, and TXT files
- Smart fallback: PyMuPDF → pdfplumber for complex PDFs
- Encoding detection (UTF-8 → latin-1 fallback)
- Stores extracted text + metadata in SQLite
- Structured logging with daily log rotation

### Stage 2 — Embedding Model Comparison
- Compares MiniLM vs MPNet vs OpenAI embeddings
- Metrics: cosine similarity, separation score, speed, dimensions
- All runs tracked and visualized in MLflow dashboard
- Evidence-based model selection

### Stage 3 — RAG Pipeline (Deep)
- Three chunking strategies: fixed, recursive, semantic
- ChromaDB vector store with persistent storage
- Hybrid search: BM25 (keyword) + vector (semantic)
- Reciprocal Rank Fusion (RRF) for result merging
- Reranking before LLM context injection
- Source citations in every answer

### Stage 4 — LLM Frameworks + Memory
- LlamaIndex VectorStoreIndex backed by ChromaDB
- LangChain conversation memory (session-based)
- Query router: automatically decides docs vs general knowledge
- Multi-turn conversations with context persistence

### Stage 5 — ReAct Agent
- Four tools: search_documents, calculate, summarize_document, answer_general
- Thought → Action → Observation loop
- Max iteration cap to prevent infinite loops
- Full reasoning trace available on demand

### Stage 6 — Deployment & Monitoring
- Dockerfile + Docker Compose (API + MLflow services)
- Prometheus metrics auto-instrumented on all endpoints
- Request timing middleware (X-Response-Time-Ms header)
- Enhanced health check with system resource stats

### Stage 7 — Guardrails & Safety
- Prompt injection detection (15+ patterns)
- Harmful content blocking
- PII detection: email, phone, credit card, Aadhaar, PAN
- PII redaction in outputs (replaces with [REDACTED:TYPE])
- Topic guardrail: keeps assistant on-scope
- Output validation: hallucination signal detection, citation check

### Stage 8 — Multimodal (Vision)
- Automatic scanned PDF detection
- Page-by-page vision LLM processing for scanned PDFs
- Direct image upload (JPG, PNG, WEBP)
- Text extraction + description generation from images
- All image content stored and searchable via RAG

### Stage 9 — GraphRAG + Knowledge Graphs
- LLM-based entity and relation extraction from chunks
- NetworkX directed graph persisted to disk
- BFS traversal up to 2 hops from query entities
- Two-path retrieval: graph traversal + vector search combined
- Graph exploration endpoint (nodes by type + all edges)

### Stage 10 — Multi-Agent System (LangGraph)
- Three specialized agents: Researcher, Analyst, Critic
- LangGraph StateGraph with conditional routing
- Critic can reject and send back to Researcher with specific feedback
- Loop until approved or max_iterations reached
- Compiler node creates clean final answer
- Full agent trace available for debugging

---

## 📡 API Endpoints

| Method | Endpoint | Stage | Description |
|---|---|---|---|
| POST | `/upload` | 1 | Upload PDF/CSV/TXT |
| GET | `/documents` | 1 | List all documents |
| POST | `/experiments/run` | 2 | Compare embedding models |
| POST | `/index/{id}` | 3 | Chunk + embed + store |
| POST | `/query` | 3 | RAG query with citations |
| POST | `/llamaindex/index` | 4 | LlamaIndex indexing |
| POST | `/chat` | 4 | Chat with memory + routing |
| POST | `/chat/reset` | 4 | Clear conversation memory |
| POST | `/agent/run` | 5 | Single ReAct agent |
| GET | `/health` | 6 | System health + resource stats |
| GET | `/metrics` | 6 | Prometheus metrics |
| POST | `/upload/image` | 8 | Upload image (vision LLM) |
| POST | `/graphrag/build/{id}` | 9 | Build knowledge graph |
| POST | `/graphrag/query` | 9 | Graph + vector retrieval |
| GET | `/graphrag/explore` | 9 | Explore graph structure |
| POST | `/multiagent/run` | 10 | 3-agent collaborative research |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here   # optional
OPENAI_API_KEY=your_openai_key_here         # optional
```

Get a free Groq API key at: https://console.groq.com

### 5. Run the backend
```bash
uvicorn backend.main:app --reload
```

API available at: `http://127.0.0.1:8000`
Interactive docs at: `http://127.0.0.1:8000/docs`

### 6. Run MLflow dashboard (optional)
```bash
mlflow ui
```
Dashboard at: `http://127.0.0.1:5000`

### 7. Run with Docker (production)
```bash
docker compose up --build
```

---

## 🧪 Quick Test Flow

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/upload -F "file=@yourfile.pdf"
# Note the document_id in response

# 2. Index it
curl -X POST "http://localhost:8000/index/1?strategy=recursive"

# 3. Ask a question
curl -X POST "http://localhost:8000/query?question=What+is+this+document+about&top_k=3"

# 4. Chat with memory
curl -X POST "http://localhost:8000/chat?question=Summarize+the+key+points&session_id=test"

# 5. Run the multi-agent system
curl -X POST "http://localhost:8000/multiagent/run?question=Analyze+the+technical+profile&show_agent_trace=true"
```

---

## 📁 Project Structure
ai-research-assistant/
├── backend/
│   ├── main.py                    # FastAPI app + all endpoints
│   ├── logger.py                  # Centralized logging
│   ├── database/
│   │   └── db.py                  # SQLite + SQLAlchemy setup
│   ├── pipeline/
│   │   ├── extractor.py           # PDF/CSV/TXT text extraction
│   │   ├── cleaner.py             # Text normalization
│   │   └── storage.py             # Document persistence
│   ├── embeddings/
│   │   ├── base.py                # Abstract embedder interface
│   │   ├── minilm.py              # MiniLM embedder
│   │   ├── mpnet.py               # MPNet embedder
│   │   └── openai_embed.py        # OpenAI embedder
│   ├── experiments/
│   │   └── compare_embeddings.py  # MLflow experiment runner
│   ├── rag/
│   │   ├── chunker.py             # Fixed/recursive/semantic chunking
│   │   ├── vector_store.py        # ChromaDB operations
│   │   ├── retriever.py           # Hybrid search + reranking
│   │   └── pipeline.py            # Full RAG pipeline
│   ├── llamaindex/
│   │   ├── loader.py              # SQLite → LlamaIndex documents
│   │   ├── indexer.py             # VectorStoreIndex builder
│   │   └── query_engine.py        # Query engine with postprocessing
│   ├── langchain/
│   │   ├── memory.py              # Session-based conversation memory
│   │   ├── router.py              # Keyword + LLM query router
│   │   └── chat_pipeline.py       # Chat with memory entry point
│   ├── agents/
│   │   ├── tools.py               # 4 agent tools
│   │   ├── agent_loop.py          # ReAct loop implementation
│   │   └── agent_pipeline.py      # Single agent entry point
│   ├── guardrails/
│   │   ├── input_guard.py         # Injection + harmful content detection
│   │   ├── pii_detector.py        # PII detection + redaction
│   │   ├── topic_guard.py         # Topic relevance check
│   │   ├── output_guard.py        # Output safety validation
│   │   └── pipeline.py            # Guards orchestrator
│   ├── multimodal/
│   │   ├── vision_extractor.py    # Vision LLM image processing
│   │   ├── pdf_scanner.py         # Scanned PDF → image → text
│   │   └── image_handler.py       # Direct image upload handler
│   ├── graphrag/
│   │   ├── extractor.py           # LLM entity/relation extraction
│   │   ├── graph_store.py         # NetworkX graph build + query
│   │   ├── graph_retriever.py     # Two-path retrieval
│   │   └── pipeline.py            # GraphRAG entry point
│   ├── multiagent/
│   │   ├── state.py               # Shared LangGraph state
│   │   ├── researcher.py          # Researcher agent node
│   │   ├── analyst.py             # Analyst agent node
│   │   ├── critic.py              # Critic agent node
│   │   ├── graph.py               # LangGraph workflow
│   │   └── pipeline.py            # Multi-agent entry point
│   └── middleware/
│       └── monitoring.py          # Request logging + system stats
├── uploads/                       # Uploaded files
├── chroma_db/                     # ChromaDB vector storage
├── graph_data/                    # Knowledge graphs (JSON)
├── logs/                          # Daily log files
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
└── .env                           # API keys (not committed)

---

## 🔐 Security Features

- Prompt injection detection and blocking
- PII detection: email, phone, credit card, Aadhaar, PAN card
- PII redaction in all LLM outputs
- Topic guardrail keeps the assistant on-scope
- Hallucination signal detection in responses
- Input length limits (max 5000 characters)
- Harmful content pattern matching

---

## 📊 Monitoring

- **Health check**: `GET /health` — system stats, DB status, resource usage
- **Prometheus metrics**: `GET /metrics` — all endpoint metrics auto-collected
- **Request timing**: Every response includes `X-Response-Time-Ms` header
- **MLflow**: All embedding experiments tracked with metrics and parameters
- **Structured logs**: Every pipeline step logged with timestamps and context

---

## 🌿 Git Branch Strategy
main (stable)
└── phase1 (integration branch)
├── stage-1/data-ingestion
├── stage-2/embedding-experiments
├── stage-3/rag-pipeline
├── stage-4/llm-frameworks
├── stage-5/agents
├── stage-6/deployment
├── stage-7/guardrails
├── stage-8/multimodal
├── stage-9/graphrag
└── stage-10/multiagent

Each stage was developed in isolation, tested, then merged — mirroring real team workflows.

---

## 🎓 Learning Outcomes

By building this project you gain hands-on experience with:

- Production data pipelines and text extraction
- Embedding model evaluation with experiment tracking
- Advanced RAG: hybrid search, chunking strategies, reranking
- LLM frameworks: LlamaIndex and LangChain
- Agentic systems: ReAct pattern, tool calling
- Docker containerization and health monitoring
- LLM safety: guardrails, PII handling, prompt injection defense
- Multimodal AI: vision LLMs for images and scanned documents
- Knowledge graphs: entity extraction and graph traversal
- Multi-agent orchestration with LangGraph

---

## 🤝 Contributing

This is a learning project. Feel free to fork it, extend it, or use it as a base for your own AI applications.

---

## 📄 License

MIT License — free to use for learning and portfolio purposes.

---

## 👨‍💻 Author

**M.S.N. Thoshanth Reddy**
B.Tech — Hyderabad Institute of Technology and Management (HITAM)

Built as part of a structured AI Engineering learning roadmap covering all core concepts needed for AI Engineer roles in 2025.

- GitHub: github.com/thoshanth_reddy
- LinkedIn: linkedin.com/in/snthoshanthreddymandapati
- Email: mthoshanthreddy@gmail.com
