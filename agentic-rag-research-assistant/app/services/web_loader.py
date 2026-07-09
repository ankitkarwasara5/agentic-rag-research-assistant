"""Web page extraction helpers."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from app.config import Settings, get_settings
from app.schemas import RawSource, SearchResult
from app.utils.text_utils import normalize_whitespace, safe_truncate, stable_id


class WebLoadError(RuntimeError):
    """Raised when a web page cannot be loaded."""


def fetch_webpage_text(
    url: str, settings: Settings | None = None
) -> tuple[str, str | None]:
    """Fetch and extract readable text and title from a web page."""

    settings = settings or get_settings()
    headers = {"User-Agent": settings.user_agent}
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=settings.web_request_timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WebLoadError(f"Could not fetch {url}: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        element.decompose()

    title = normalize_whitespace(soup.title.string if soup.title else "")
    text = normalize_whitespace(soup.get_text(" "))
    return text, title or None


def search_result_to_source(
    result: SearchResult, settings: Settings | None = None
) -> RawSource:
    """Convert a search result into a RawSource, falling back to snippet text."""

    try:
        text, page_title = fetch_webpage_text(result.url, settings=settings)
    except WebLoadError:
        text, page_title = result.snippet, None

    text = normalize_whitespace(text or result.snippet)
    if not text:
        text = f"{result.title}. {result.snippet}"

    return RawSource(
        source_id=stable_id(f"web:{result.url}"),
        source_type="web",
        title=safe_truncate(page_title or result.title or result.url, 180),
        url=result.url,
        text=text,
    )

