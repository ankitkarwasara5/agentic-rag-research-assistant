"""Text parsing and deterministic fallback helpers."""

from __future__ import annotations

import hashlib
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]*")
CITATION_RE = re.compile(r"\[(Source\s+\d+)\]", re.IGNORECASE)


def stable_id(value: str, length: int = 16) -> str:
    """Return a stable identifier derived from text."""

    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def normalize_whitespace(text: str) -> str:
    """Normalize line endings while preserving paragraph breaks."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if cleaned and not blank:
                cleaned.append("")
            blank = True
            continue
        cleaned.append(line)
        blank = False
    return "\n".join(cleaned).strip()


def safe_truncate(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary when possible."""

    if len(text) <= max_chars:
        return text
    shortened = text[: max(0, max_chars - 1)].rsplit(" ", 1)[0].rstrip()
    return f"{shortened}…"


def parse_list(text: str, fallback: list[str], limit: int) -> list[str]:
    """Parse bullet/numbered model output into a compact list."""

    items: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
        if len(line) >= 3 and line not in items:
            items.append(line)
        if len(items) >= limit:
            break
    return items or fallback[:limit]


def tokens(text: str) -> list[str]:
    """Return normalized word-like tokens."""

    return [token.lower() for token in WORD_RE.findall(text)]


def significant_terms(text: str) -> set[str]:
    """Return non-trivial terms for lexical similarity."""

    return {token for token in tokens(text) if len(token) > 2}


def term_overlap_score(query: str, text: str) -> float:
    """Compute a bounded lexical overlap score."""

    query_terms = significant_terms(query)
    if not query_terms:
        return 0.0
    text_counts = Counter(tokens(text))
    matched = sum(min(2, text_counts.get(term, 0)) for term in query_terms)
    score = matched / max(1, len(query_terms) * 2)
    return round(min(1.0, score), 4)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into deterministic overlapping character chunks."""

    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if end < len(text):
            boundary = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind(" "))
            if boundary > chunk_size // 2:
                end = start + boundary + 1
                chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + step, end - overlap)
    return chunks


def extract_citation_ids(text: str) -> list[str]:
    """Extract normalized citation labels from report text."""

    found = CITATION_RE.findall(text)
    return [re.sub(r"\s+", " ", item.title()) for item in found]
