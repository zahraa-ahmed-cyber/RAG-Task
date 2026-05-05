"""app/api/chat.py — /chat endpoint."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from app.config import get_settings
from app.rag.pipeline import chat as rag_chat, chat_stream, retrieve_for_chat
from app.models.inference import generate as finetuned_generate

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── schemas ───────────────────────────────────────────────────────────────────

class ChatTurn(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User question")
    history: list[ChatTurn] | None = Field(None, description="Previous conversation turns")
    stream: bool = Field(False, description="Enable token-by-token streaming")
    k: int | None = Field(
        None,
        ge=1,
        le=30,
        description="Override number of chunks to retrieve (default: TOP_K_RETRIEVAL from .env)",
    )


class ChatResponse(BaseModel):
    response: str
    sources: list[str]


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse, summary="Ask a question")
async def chat_endpoint(body: ChatRequest):
    """
    Send a message to the RAG chatbot.

    - Retrieves relevant context from the vector store
    - Generates an answer grounded in that context
    - Returns the answer and the source document names
    """
    cfg = get_settings()
    history = [t.model_dump() for t in body.history] if body.history else None

    if body.stream:
        if cfg.use_finetuned_model:
            raise HTTPException(
                status_code=400,
                detail="Streaming is only available in RAG mode. Disable USE_FINETUNED_MODEL for streaming.",
            )
        # Streaming response (Server-Sent Events)
        async def _stream():
            try:
                async for token in chat_stream(body.message, k=body.k, history=history):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error(f"Streaming error: {exc}")
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # Normal synchronous response
    try:
        if cfg.use_finetuned_model:
            # Hybrid behavior: still retrieve context and feed it to the fine-tuned model.
            context, sources = retrieve_for_chat(body.message, history, body.k)
            answer = finetuned_generate(body.message, context=context)
            return ChatResponse(response=answer, sources=sources)

        result = rag_chat(body.message, history=history, k=body.k)
        return ChatResponse(**result)
    except RuntimeError as exc:
        logger.error(f"Chat error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Unexpected error in /chat: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
