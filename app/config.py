"""Central configuration — reads from .env / environment variables."""

from __future__ import annotations
from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str = Field(
        default="dummy-key",
        validation_alias=AliasChoices("OPENAI_API_KEY", "GROQ_API_KEY"),
    )
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_provider: str = "local"       # "local" | "openai"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    vector_store_path: str = "./data/vectorstore"
    vector_store_type: str = "faiss"        # "faiss" | "chroma"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 12

    # Fine-tuned model
    finetuned_model_path: str = "./data/finetuned_model"
    use_finetuned_model: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
