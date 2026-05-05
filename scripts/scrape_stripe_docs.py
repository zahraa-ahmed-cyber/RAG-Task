"""
scripts/scrape_stripe_docs.py
─────────────────────────────
Scrapes Stripe's public documentation pages and saves each page as a
plain-text (.txt) file under  data/raw/stripe/.

Usage:
    python scripts/scrape_stripe_docs.py

No API key required — only public pages are fetched.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

# ── Target pages ──────────────────────────────────────────────────────────────
STRIPE_PAGES: list[tuple[str, str]] = [
    ("payments_overview",    "https://stripe.com/docs/payments"),
    ("checkout",             "https://stripe.com/docs/payments/checkout"),
    ("invoicing",            "https://stripe.com/docs/invoicing"),
    ("subscriptions",        "https://stripe.com/docs/billing/subscriptions/overview"),
    ("refunds",              "https://stripe.com/docs/refunds"),
    ("disputes",             "https://stripe.com/docs/disputes"),
    ("radar_fraud",          "https://stripe.com/docs/radar"),
    ("webhooks",             "https://stripe.com/docs/webhooks"),
    ("api_keys",             "https://stripe.com/docs/keys"),
    ("testing",              "https://stripe.com/docs/testing"),
    ("connect_overview",     "https://stripe.com/docs/connect"),
    ("payouts",              "https://stripe.com/docs/payouts"),
    ("tax_overview",         "https://stripe.com/docs/tax"),
    ("customers",            "https://stripe.com/docs/customers"),
    ("payment_methods",      "https://stripe.com/docs/payments/payment-methods/overview"),
]

OUT_DIR = Path("data/raw/stripe")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAG-Bot/1.0; "
        "+https://github.com/your-org/chatbot-rag)"
    )
}


def _clean(soup: BeautifulSoup) -> str:
    """Extract meaningful text from a Stripe docs page."""
    # Remove nav, footer, code blocks (keep prose)
    for tag in soup.select("nav, footer, script, style, [role='navigation']"):
        tag.decompose()

    # Main content lives in <main> or article-like divs
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return ""

    text = main.get_text(separator="\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape(pages: list[tuple[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    for name, url in pages:
        dest = out_dir / f"{name}.txt"
        if dest.exists():
            logger.info(f"[skip] {name} already exists")
            continue
        try:
            logger.info(f"[fetch] {url}")
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            text = _clean(soup)
            if len(text) < 200:
                logger.warning(f"  ↳ very short content ({len(text)} chars) — skipping")
                continue
            dest.write_text(text, encoding="utf-8")
            logger.success(f"  ↳ saved {len(text):,} chars → {dest}")
        except Exception as exc:
            logger.error(f"  ↳ failed: {exc}")
        time.sleep(1.2)   # polite crawl delay


if __name__ == "__main__":
    scrape(STRIPE_PAGES, OUT_DIR)
    logger.info("Done.")
