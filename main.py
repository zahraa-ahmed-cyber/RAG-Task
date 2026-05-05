"""main.py — FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.api.chat   import router as chat_router
from app.api.ingest import router as ingest_router
from app.api.search import router as search_router


# ── startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    logger.info("=" * 55)
    logger.info("  Chatbot RAG — starting up")
    logger.info(f"  Embedding : {cfg.embedding_provider} / {cfg.embedding_model}")
    logger.info(f"  LLM       : {cfg.llm_model}")
    logger.info(f"  VectorDB  : {cfg.vector_store_type}  →  {cfg.vector_store_path}")
    logger.info("=" * 55)

    # Pre-build index if raw data exists but no vector store yet
    store_path = Path(cfg.vector_store_path)
    data_dir   = Path("data/raw")

    if not store_path.exists() and data_dir.exists():
        logger.info("No vector store found — building from data/raw/ …")
        try:
            from app.rag.document_loader import load_and_chunk
            from app.rag.vector_store    import get_or_build_store
            chunks = load_and_chunk(data_dir)
            if chunks:
                get_or_build_store(chunks)
                logger.success(f"Index built with {len(chunks)} chunks.")
            else:
                logger.warning("No documents found in data/raw/ — index not built.")
        except Exception as exc:
            logger.error(f"Could not build index on startup: {exc}")

    yield
    logger.info("Chatbot RAG — shutting down")


# ── app factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="Chatbot RAG API",
        description=(
            "AI support chatbot powered by RAG (Retrieval-Augmented Generation) "
            "on Stripe documentation + Bitext customer-support data."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(ingest_router)
    app.include_router(search_router)

    return app


app = create_app()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=True,
        log_level=cfg.log_level.lower(),
    )
