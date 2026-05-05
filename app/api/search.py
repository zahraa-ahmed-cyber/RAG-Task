"""app/api/search.py — /search and /health utility endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.rag.vector_store import similarity_search

router = APIRouter(tags=["Utility"])


class SearchResult(BaseModel):
    content: str
    source: str
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@router.get("/search", response_model=SearchResponse, summary="Raw vector search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(4, ge=1, le=20, description="Number of results"),
):
    """Search the vector store directly — useful for debugging retrieval quality."""
    try:
        docs = similarity_search(q, k=k)
        results = [
            SearchResult(
                content=d.page_content[:500],
                source=d.metadata.get("filename", d.metadata.get("source", "?")),
            )
            for d in docs
        ]
        return SearchResponse(query=q, results=results)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}
