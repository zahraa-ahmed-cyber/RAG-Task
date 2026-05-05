"""app/rag — RAG components."""
from app.rag.document_loader import load_and_chunk, load_single_file
from app.rag.vector_store    import get_or_build_store, add_documents, similarity_search
try:
    from app.rag.pipeline import chat, chat_stream
except Exception:
    # Keep package import lightweight for scripts that only need loaders/stores.
    # The API/runtime path imports pipeline directly when required.
    chat = None
    chat_stream = None

__all__ = [
    "load_and_chunk", "load_single_file",
    "get_or_build_store", "add_documents", "similarity_search",
    "chat", "chat_stream",
]
