"""app/api/ingest.py — /ingest endpoint for uploading new documents."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from loguru import logger

from app.rag.document_loader import load_single_file
from app.rag.vector_store import add_documents

router = APIRouter(prefix="/ingest", tags=["Ingest"])

_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}


class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int


@router.post("", response_model=IngestResponse, summary="Upload a document")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, MD, HTML) to extend the knowledge base.

    The document is:
    1. Parsed and cleaned
    2. Split into overlapping chunks
    3. Embedded and added to the vector store
    """
    from pathlib import Path
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        logger.info(f"Ingesting {file.filename!r}  ({len(content):,} bytes)")
        chunks = load_single_file(content, file.filename or "upload")
        add_documents(chunks)

        logger.success(f"Ingested {len(chunks)} chunks from {file.filename!r}")
        return IngestResponse(
            message="Document ingested successfully.",
            filename=file.filename or "",
            chunks_added=len(chunks),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Ingest error for {file.filename!r}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
