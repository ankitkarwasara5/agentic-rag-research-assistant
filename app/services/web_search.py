"""Free web search and page extraction."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import Settings, get_settings
from app.schemas import RawSource, SearchResult
from app.utils.text import normalize_whitespace, stable_id


class WebSearchError(RuntimeError):
    """Raised when search cannot run."""


def _ddgs_class():
    try:
        from ddgs import DDGS
    except ImportError as exc:  # pragma: no cover
        raise WebSearchError("Install `ddgs` to enable web research.") from exc
    return DDGS


def search_web(
    questions: list[str],
    settings: Settings | None = None,
) -> list[SearchResult]:
    """Search DuckDuckGo and deduplicate URLs."""

    settings = settings or get_settings()
    results: list[SearchResult] = []
    seen: set[str] = set()
    DDGS = _ddgs_class()
    try:
        with DDGS() as client:
            for question in questions:
                for item in client.text(
                    question,
                    max_results=settings.web_max_results,
                    safesearch="moderate",
                ):
                    url = str(item.get("href") or item.get("url") or "").strip()
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    results.append(
                        SearchResult(
                            title=str(item.get("title") or url),
                            url=url,
                            snippet=str(item.get("body") or item.get("snippet") or ""),
                        )
                    )
                    if len(results) >= settings.web_max_results:
                        return results
    except Exception as exc:  # noqa: BLE001
        raise WebSearchError(f"DuckDuckGo search failed: {exc}") from exc
    return results


def _safe_public_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )


def result_to_source(
    result: SearchResult,
    settings: Settings | None = None,
) -> RawSource:
    """Fetch readable page text, falling back to the search snippet."""

    settings = settings or get_settings()
    text = result.snippet
    if _safe_public_url(result.url):
        try:
            response = httpx.get(
                result.url,
                timeout=settings.web_request_timeout_seconds,
                follow_redirects=False,
                headers={"User-Agent": "AgenticRAGResearchAssistant/2.0"},
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type:
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "noscript", "svg"]):
                    tag.decompose()
                extracted = normalize_whitespace(soup.get_text("\n"))
                if extracted:
                    text = extracted[: settings.max_source_chars]
        except Exception:
            pass

    return RawSource(
        source_id=stable_id(f"web:{result.url}"),
        source_type="web",
        title=result.title,
        url=result.url,
        text=normalize_whitespace(text)[: settings.max_source_chars],
    )
