"""
scripts/evaluate.py
────────────────────
Evaluate RAG accuracy against a small hand-crafted Q&A test set.

Metrics:
  • Answer relevance  — does the answer mention key expected terms?
  • Source hit rate   — was the expected source retrieved?
  • "I don't know" rate — how often does the model correctly abstain?

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from app.rag.pipeline import chat, retrieve

# ── test set ──────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "question": "How do I issue a refund on Stripe?",
        "expected_keywords": ["refund", "dashboard", "charge"],
        "expected_source_hint": "refund",
    },
    {
        "question": "What is Stripe Radar used for?",
        "expected_keywords": ["fraud", "radar", "risk"],
        "expected_source_hint": "radar",
    },
    {
        "question": "How do I configure webhook endpoints and verify signatures?",
        "expected_keywords": ["webhook", "endpoint", "signature"],
        "expected_source_hint": "webhook",
    },
    {
        "question": "What are the test card numbers for Stripe?",
        "expected_keywords": ["4242", "test", "card"],
        "expected_source_hint": "testing",
    },
    {
        "question": "How do I cancel a subscription?",
        "expected_keywords": ["subscription", "cancel"],
        "expected_source_hint": "subscription",
    },
    {
        "question": "What is the meaning of life?",          # out-of-scope
        "expected_keywords": ["don't", "information", "not"],
        "expected_source_hint": None,   # shouldn't hallucinate
    },
]


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate() -> None:
    results = []
    for tc in TEST_CASES:
        q = tc["question"]
        logger.info(f"\n▸ {q}")

        try:
            result = chat(q)
            answer  = result["response"].lower()
            sources = [s.lower() for s in result["sources"]]

            kw_hit = sum(1 for kw in tc["expected_keywords"] if kw.lower() in answer)
            kw_rate = kw_hit / len(tc["expected_keywords"])

            src_hint = tc["expected_source_hint"]
            src_hit = any(src_hint in s for s in sources) if src_hint else True

            logger.info(f"  Answer      : {result['response'][:120]} …")
            logger.info(f"  Sources     : {result['sources']}")
            logger.info(f"  Keyword hit : {kw_hit}/{len(tc['expected_keywords'])}  ({kw_rate:.0%})")
            logger.info(f"  Source hit  : {src_hit}")

            results.append({
                "question":    q,
                "kw_rate":     kw_rate,
                "source_hit":  src_hit,
            })
        except Exception as exc:
            logger.error(f"  Error: {exc}")
            results.append({"question": q, "kw_rate": 0, "source_hit": False})

    # Summary
    avg_kw  = sum(r["kw_rate"]  for r in results) / len(results)
    avg_src = sum(r["source_hit"] for r in results) / len(results)
    logger.info("\n" + "=" * 50)
    logger.info(f"  Avg keyword hit rate : {avg_kw:.1%}")
    logger.info(f"  Avg source hit rate  : {avg_src:.1%}")
    logger.info("=" * 50)


if __name__ == "__main__":
    evaluate()
