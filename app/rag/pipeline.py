"""app/rag/pipeline.py — the core RAG chain."""

from __future__ import annotations

from typing import AsyncIterator

try:
    # LangChain >= 0.3
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    # LangChain < 0.3
    from langchain.prompts import ChatPromptTemplate
try:
    # LangChain >= 0.3
    from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
except ImportError:
    # LangChain < 0.3
    from langchain.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
try:
    # LangChain >= 0.3
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    # LangChain < 0.3
    from langchain.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import get_settings
from app.rag.vector_store import similarity_search

# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful customer support assistant for a SaaS payment platform.

You receive:
- **Context**: excerpts from internal documentation (retrieved for this turn).
- **Prior conversation** (if any): earlier user and assistant turns in this chat.

**How to answer**

1. **New factual questions** (products, Stripe, policies, or troubleshooting that needs documentation): base the answer **only** on **Context**. If the answer is not in Context, say exactly: "I don't have enough information to answer that question."

2. **Conversation follow-ups** — use **prior assistant messages** (and user messages) as the primary source; Context is optional supporting detail:
   - Summarize, shorter, rephrase, bullet points, numbered or checklist-style steps
   - Explain in simpler or **non-technical** language
   - What to try **first** if something failed, or next steps implied by the **last** assistant reply
   - Clarifications about what was **just** said ("what did you mean by…")
   Do **not** refuse these only because the follow-up sentence is not literally inside Context. Still do **not** invent specific prices, legal guarantees, or policy details beyond what appears in Context or prior assistant text.

3. **"Which source files / documents did you use?"** — list the filenames under **Retrieved documents for this turn** below. Do not invent file names.

4. Be concise, friendly, and professional.

Retrieved documents for this turn: {source_names}

Context:
{context}
"""

HUMAN_TEMPLATE = "{question}"

_META_KEYS = (
    "summarize", "summary", "bullet", "checklist", "exact steps",
    "step-by-step", "step by step", "what should i do", "if that fails",
    "if this fails", "non-technical", "non technical", "layman", "plain english",
    "simpler", "rephrase", "reword", "shorter", "briefly", "in 3 ", " three ",
    " which source", " which file", " what files", " what documents",
    "where did you", "cite your", "sources did", "restate", "clarify",
)


def _looks_like_meta_followup(message: str) -> bool:
    m = f" {message.lower()} "
    return any(k in m for k in _META_KEYS)


def _retrieval_query(message: str, history: list[dict] | None) -> str:
    """Expand retrieval for vague or meta follow-ups so chunks stay on-topic."""
    message = message.strip()
    if not history:
        return message
    last = history[-1]
    u = str(last.get("user", "")).strip()
    a = str(last.get("assistant", "")).strip()
    short = len(message) < 55
    if _looks_like_meta_followup(message) or short:
        return f"{u}\n{a}\n{message}".strip()[:6000]
    return message


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm(streaming: bool = False) -> ChatOpenAI:
    cfg = get_settings()
    kwargs: dict = dict(
        model=cfg.llm_model,
        openai_api_key=cfg.openai_api_key,
        temperature=0.2,
        streaming=streaming,
    )
    if cfg.openai_base_url:
        kwargs["openai_api_base"] = cfg.openai_base_url
    return ChatOpenAI(**kwargs)


# ── retriever helper ──────────────────────────────────────────────────────────

def retrieve(query: str, k: int | None = None) -> tuple[str, list[str]]:
    """Return (context_string, list_of_source_names).

    *k* overrides settings TOP_K_RETRIEVAL for this retrieval only.
    """
    docs = similarity_search(query, k=k)
    sources = list({d.metadata.get("filename", d.metadata.get("source", "unknown")) for d in docs})
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return context, sources


def retrieve_for_chat(message: str, history: list[dict] | None, k: int | None = None) -> tuple[str, list[str]]:
    """Retrieve using an expanded query when the user message is a short or meta follow-up."""
    return retrieve(_retrieval_query(message, history), k=k)


def _build_rag_messages(
    message: str,
    history: list[dict] | None,
    k: int | None,
) -> tuple[list[BaseMessage], list[str]]:
    rq = _retrieval_query(message, history)
    context, sources = retrieve(rq, k=k)
    source_names = ", ".join(sources) if sources else "(none)"
    system_content = SYSTEM_PROMPT.format(context=context, source_names=source_names)
    messages: list[BaseMessage] = [SystemMessage(content=system_content)]
    if history:
        for turn in history[-6:]:
            messages.append(HumanMessage(content=turn["user"]))
            messages.append(AIMessage(content=turn["assistant"]))
    messages.append(HumanMessage(content=message))
    return messages, sources


# ── chain ─────────────────────────────────────────────────────────────────────

def _build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_TEMPLATE),
    ])
    return prompt | _get_llm() | StrOutputParser()


# ── public API ────────────────────────────────────────────────────────────────

def chat(message: str, history: list[dict] | None = None, k: int | None = None) -> dict:
    """
    Synchronous RAG chat.

    Returns:
        {"response": str, "sources": list[str]}
    """
    messages, sources = _build_rag_messages(message, history, k)
    logger.debug(f"Retrieved {len(sources)} source(s) for: {message[:60]!r}")

    llm = _get_llm()
    response: BaseMessage = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    return {"response": answer, "sources": sources}


async def chat_stream(
    message: str,
    k: int | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """
    Async streaming RAG chat — yields text tokens.
    """
    messages, _sources = _build_rag_messages(message, history, k)

    llm = _get_llm(streaming=True)
    async for chunk in llm.astream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            yield token
