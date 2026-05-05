"""app/rag/embeddings.py — embedding provider factory."""

from __future__ import annotations

from functools import lru_cache

from langchain.embeddings.base import Embeddings
from loguru import logger

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    cfg = get_settings()

    if cfg.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Using OpenAI embeddings: {cfg.embedding_model}")
        return OpenAIEmbeddings(
            model=cfg.embedding_model,
            openai_api_key=cfg.openai_api_key,
        )

    # Default: local sentence-transformers (free, no API key needed)
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info(f"Using local HuggingFace embeddings: {cfg.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
