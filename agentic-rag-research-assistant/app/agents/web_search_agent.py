"""Web Search Agent using free DuckDuckGo search packages."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.schemas import SearchResult


class WebSearchError(RuntimeError):
    """Raised when DuckDuckGo search cannot run."""


def _load_ddgs_class():
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS

            return DDGS
        except ImportError as exc:
            raise WebSearchError(
                "Install `ddgs` or `duckduckgo-search` to enable web search."
            ) from exc


def search_web(
    questions: list[str],
    settings: Settings | None = None,
) -> list[SearchResult]:
    """Search DuckDuckGo for each refined question and deduplicate URLs."""

    settings = settings or get_settings()
    DDGS = _load_ddgs_class()
    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    ddgs = DDGS()
    try:
        for question in questions:
            for item in ddgs.text(
                question,
                max_results=settings.web_max_results,
                safesearch="moderate",
            ):
                url = item.get("href") or item.get("url") or ""
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(
                    SearchResult(
                        title=item.get("title") or url,
                        url=url,
                        snippet=item.get("body") or item.get("snippet") or "",
                    )
                )
                if len(results) >= settings.web_max_results:
                    return results
    except Exception as exc:  # noqa: BLE001
        raise WebSearchError(f"DuckDuckGo search failed: {exc}") from exc
    finally:
        close = getattr(ddgs, "close", None)
        if callable(close):
            close()

    return results
