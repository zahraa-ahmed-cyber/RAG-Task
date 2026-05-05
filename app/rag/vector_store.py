"""app/rag/vector_store.py — build, persist and query the vector store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
try:
    # LangChain >= 0.3
    from langchain_core.documents import Document
except ImportError:
    # LangChain < 0.3
    from langchain.schema import Document
try:
    # LangChain >= 0.3
    from langchain_core.vectorstores import VectorStore
except ImportError:
    # LangChain < 0.3
    from langchain.vectorstores.base import VectorStore
from loguru import logger

from app.config import get_settings
from app.rag.embeddings import get_embeddings

if TYPE_CHECKING:
    pass


# ── internal helpers ──────────────────────────────────────────────────────────

def _store_path() -> Path:
    return Path(get_settings().vector_store_path)


def _build_faiss(docs: list[Document]) -> VectorStore:
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(docs, get_embeddings())


def _load_faiss() -> VectorStore | None:
    from langchain_community.vectorstores import FAISS
    path = _store_path()
    if not path.exists():
        return None
    try:
        store = FAISS.load_local(
            str(path),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Loaded FAISS store from {path}")
        return store
    except Exception as exc:
        logger.warning(f"Could not load FAISS store: {exc}")
        return None


def _save_faiss(store: VectorStore) -> None:
    from langchain_community.vectorstores import FAISS
    path = _store_path()
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))          # type: ignore[attr-defined]
    logger.info(f"FAISS store saved → {path}")


def _build_chroma(docs: list[Document]) -> VectorStore:
    from langchain_community.vectorstores import Chroma
    path = _store_path()
    path.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        docs,
        get_embeddings(),
        persist_directory=str(path),
    )


def _load_chroma() -> VectorStore | None:
    from langchain_community.vectorstores import Chroma
    path = _store_path()
    if not path.exists():
        return None
    try:
        store = Chroma(
            embedding_function=get_embeddings(),
            persist_directory=str(path),
        )
        logger.info(f"Loaded Chroma store from {path}")
        return store
    except Exception as exc:
        logger.warning(f"Could not load Chroma store: {exc}")
        return None


# ── public API ────────────────────────────────────────────────────────────────

_store_cache: VectorStore | None = None


def get_or_build_store(docs: list[Document] | None = None) -> VectorStore:
    """Return the cached store, loading from disk if available, else building."""
    global _store_cache

    cfg = get_settings()

    if _store_cache is not None:
        return _store_cache

    # Try to load existing
    if cfg.vector_store_type == "chroma":
        _store_cache = _load_chroma()
    else:
        _store_cache = _load_faiss()

    if _store_cache is not None:
        return _store_cache

    # Build from documents
    if not docs:
        raise RuntimeError(
            "No vector store on disk and no documents provided to build one. "
            "Run the /ingest endpoint or the build_index.py script first."
        )

    logger.info(f"Building new {cfg.vector_store_type.upper()} store with {len(docs)} chunks …")
    if cfg.vector_store_type == "chroma":
        _store_cache = _build_chroma(docs)
    else:
        _store_cache = _build_faiss(docs)
        _save_faiss(_store_cache)

    return _store_cache


def add_documents(docs: list[Document]) -> None:
    """Add new documents to the existing store (or build one)."""
    global _store_cache
    cfg = get_settings()

    if _store_cache is None:
        _store_cache = get_or_build_store(docs)
        return

    _store_cache.add_documents(docs)      # type: ignore[arg-type]

    if cfg.vector_store_type == "faiss":
        _save_faiss(_store_cache)

    logger.info(f"Added {len(docs)} chunks to the store")


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    cfg = get_settings()
    store = get_or_build_store()
    k = k or cfg.top_k_retrieval
    return store.similarity_search(query, k=k)


def reset_store() -> None:
    """Clear the in-memory cache (useful for testing)."""
    global _store_cache
    _store_cache = None
