"""app/rag/document_loader.py — load & chunk documents from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

try:
    # LangChain >= 0.3
    from langchain_core.documents import Document
except ImportError:
    # LangChain < 0.3
    from langchain.schema import Document

try:
    # Modern split package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Older LangChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import get_settings

_SUPPORTED = {".txt", ".pdf", ".html", ".htm", ".md"}


# ── low-level loaders ─────────────────────────────────────────────────────────

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    try:
        import pdfplumber
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as exc:
        logger.warning(f"pdfplumber failed for {path.name}: {exc} — falling back to pypdf")
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


_LOADER_MAP = {
    ".txt": _load_txt,
    ".md":  _load_txt,
    ".pdf": _load_pdf,
    ".html": _load_html,
    ".htm":  _load_html,
}


# ── public API ────────────────────────────────────────────────────────────────

def iter_raw_documents(data_dir: str | Path) -> Iterator[Document]:
    """Yield one Document per file in *data_dir* (recursive)."""
    root = Path(data_dir)
    if not root.exists():
        logger.warning(f"Data directory not found: {root}")
        return

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in _SUPPORTED:
            continue
        loader = _LOADER_MAP.get(path.suffix.lower())
        if loader is None:
            continue
        try:
            text = loader(path)
            if len(text.strip()) < 50:
                continue
            yield Document(
                page_content=text,
                metadata={"source": str(path), "filename": path.name},
            )
            logger.debug(f"Loaded {path.name}  ({len(text):,} chars)")
        except Exception as exc:
            logger.error(f"Could not load {path.name}: {exc}")


def load_and_chunk(data_dir: str | Path) -> list[Document]:
    """Load all documents and split into chunks."""
    cfg = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    raw = list(iter_raw_documents(data_dir))
    logger.info(f"Loaded {len(raw)} document(s) from {data_dir}")

    chunks = splitter.split_documents(raw)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def load_single_file(content: bytes, filename: str) -> list[Document]:
    """Load a single uploaded file (bytes) and chunk it."""
    cfg = get_settings()
    suffix = Path(filename).suffix.lower()

    # write to temp file so loaders can open it
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        loader = _LOADER_MAP.get(suffix)
        if loader is None:
            raise ValueError(f"Unsupported file type: {suffix}")
        text = loader(tmp_path)
    finally:
        os.unlink(tmp_path)

    doc = Document(
        page_content=text,
        metadata={"source": filename, "filename": filename},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    return splitter.split_documents([doc])
