"""Text cleaning, parsing, and chunking helpers."""

from __future__ import annotations

import re
from collections import Counter
from hashlib import sha1


_WHITESPACE_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing spaces."""

    return _WHITESPACE_RE.sub(" ", text or "").strip()


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks that some local models emit."""

    return _THINK_RE.sub("", text or "").strip()


def safe_truncate(text: str, max_chars: int) -> str:
    """Truncate text without cutting through too much context."""

    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def stable_id(value: str, prefix: str = "src") -> str:
    """Create a deterministic short id for a source-like value."""

    digest = sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def parse_list(text: str, fallback: list[str] | None = None, limit: int = 8) -> list[str]:
    """Parse bullets or numbered lines from an LLM response."""

    items: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[\).]\s+", "", line)
        if line and len(line) > 2:
            items.append(line)
    if not items and fallback:
        items = fallback
    return items[:limit]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character chunks."""

    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = end - overlap
    return [chunk for chunk in chunks if chunk]


def term_overlap_score(query: str, text: str) -> float:
    """Simple lexical score used when vector retrieval is unavailable."""

    query_terms = [
        token
        for token in re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
        if token not in {"the", "and", "for", "with", "from", "that", "this"}
    ]
    if not query_terms:
        return 0.0
    text_terms = Counter(re.findall(r"[a-zA-Z0-9]{3,}", text.lower()))
    hits = sum(text_terms.get(term, 0) for term in set(query_terms))
    return min(1.0, hits / max(1, len(set(query_terms))))

