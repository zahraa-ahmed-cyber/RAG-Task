"""
scripts/build_index.py
──────────────────────
Standalone script to build (or rebuild) the vector index from all
documents inside  data/raw/.

Usage:
    python scripts/build_index.py [--data data/raw] [--reset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from app.rag.document_loader import load_and_chunk
from app.rag.vector_store    import get_or_build_store, reset_store


def main(data_dir: str = "data/raw", reset: bool = False) -> None:
    if reset:
        logger.info("Resetting vector store cache …")
        reset_store()

    logger.info(f"Loading documents from: {data_dir}")
    chunks = load_and_chunk(data_dir)

    if not chunks:
        logger.error("No documents found. Run the scraper / dataset scripts first.")
        sys.exit(1)

    logger.info(f"Building vector store with {len(chunks)} chunks …")
    store = get_or_build_store(chunks)
    logger.success(f"Vector store ready  ({len(chunks)} chunks)")

    # Quick sanity check
    test_q = "How do I issue a refund?"
    results = store.similarity_search(test_q, k=2)
    logger.info(f"\nSanity check — query: {test_q!r}")
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("filename", "?")
        logger.info(f"  [{i}] {src}: {doc.page_content[:120]!r} …")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="data/raw",  help="Directory containing raw docs")
    parser.add_argument("--reset", action="store_true", help="Delete cached store first")
    args = parser.parse_args()
    main(args.data, args.reset)
