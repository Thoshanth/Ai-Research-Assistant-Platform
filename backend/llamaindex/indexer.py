import chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Document
import os
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("llamaindex.indexer")

CHROMA_PATH = Path("chroma_db")

# Configure LlamaIndex global settings
# This tells LlamaIndex which embedder and LLM to use everywhere
# without passing them explicitly every time

Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"  # same model from Stage 2
)

Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

Settings.chunk_size = 200       # words per chunk
Settings.chunk_overlap = 20     # overlap between chunks


def get_chroma_collection(collection_name: str = "llamaindex_docs"):
    """
    Gets or creates a ChromaDB collection specifically for LlamaIndex.
    We use a separate collection from Stage 3 so they don't interfere.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.debug(f"ChromaDB collection '{collection_name}' ready | count={collection.count()}")
    return client, collection


def build_index(documents: list[Document]) -> VectorStoreIndex:
    """
    Takes LlamaIndex Documents and builds a VectorStoreIndex.

    Internally this:
    1. Chunks each document into nodes (using Settings.chunk_size)
    2. Embeds each node using Settings.embed_model (MiniLM)
    3. Stores embeddings in ChromaDB
    4. Returns an index object you can query instantly
    """
    logger.info(f"Building LlamaIndex VectorStoreIndex | docs={len(documents)}")

    client, collection = get_chroma_collection()

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("VectorStoreIndex built and stored in ChromaDB successfully")
    return index


def load_existing_index() -> VectorStoreIndex:
    """
    Loads an already-built index from ChromaDB without re-indexing.
    Use this for querying when index already exists.
    """
    logger.info("Loading existing index from ChromaDB")

    client, collection = get_chroma_collection()

    if collection.count() == 0:
        raise ValueError("No documents indexed yet. Call build_index first.")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    logger.info(f"Index loaded | vectors in collection={collection.count()}")
    return index