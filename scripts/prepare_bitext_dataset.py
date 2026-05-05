"""
scripts/prepare_bitext_dataset.py
──────────────────────────────────
1. Downloads the Bitext Customer-Support dataset from Hugging Face.
2. Samples up to MAX_SAMPLES rows (stratified by intent).
3. Saves two artefacts:
   • data/processed/finetune_pairs.jsonl  → fine-tuning (instruction / response)
   • data/raw/bitext/                     → .txt files per intent for RAG ingestion
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from loguru import logger

MAX_SAMPLES = 1000       # total Q&A pairs kept
SEED        = 42

OUT_JSONL   = Path("data/processed/finetune_pairs.jsonl")
OUT_RAW_DIR = Path("data/raw/bitext")


# ── helpers ───────────────────────────────────────────────────────────────────

def _stratified_sample(rows: list[dict], n: int, key: str) -> list[dict]:
    """Sample n rows, keeping intent distribution proportional."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)

    total = len(rows)
    selected: list[dict] = []
    for intent, items in buckets.items():
        quota = max(1, round(n * len(items) / total))
        selected.extend(random.sample(items, min(quota, len(items))))

    # top-up / trim to exactly n
    random.shuffle(selected)
    return selected[:n]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run:  pip install datasets")

    random.seed(SEED)

    logger.info("Downloading bitext/Bitext-customer-support-llm-chatbot-training-dataset …")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
        trust_remote_code=True,
    )

    rows = [dict(r) for r in ds]
    logger.info(f"Total rows: {len(rows):,}  |  intents: {len(set(r['intent'] for r in rows))}")

    sampled = _stratified_sample(rows, MAX_SAMPLES, "intent")
    logger.info(f"Sampled {len(sampled):,} rows")

    # ── 1. Fine-tuning JSONL ──────────────────────────────────────────────────
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for r in sampled:
            fh.write(json.dumps({
                "instruction": r["instruction"],
                "response":    r["response"],
                "intent":      r["intent"],
                "category":    r.get("category", ""),
            }, ensure_ascii=False) + "\n")
    logger.success(f"Fine-tune pairs → {OUT_JSONL}  ({len(sampled)} rows)")

    # ── 2. RAG text files (one per intent) ───────────────────────────────────
    OUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    intent_docs: dict[str, list[str]] = defaultdict(list)
    for r in rows:                      # use all rows for richer RAG content
        intent_docs[r["intent"]].append(
            f"Q: {r['instruction']}\nA: {r['response']}"
        )

    for intent, qas in intent_docs.items():
        safe_name = intent.replace(" ", "_").replace("/", "-")
        dest = OUT_RAW_DIR / f"{safe_name}.txt"
        dest.write_text("\n\n---\n\n".join(qas), encoding="utf-8")

    logger.success(f"RAG text files → {OUT_RAW_DIR}  ({len(intent_docs)} files)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
