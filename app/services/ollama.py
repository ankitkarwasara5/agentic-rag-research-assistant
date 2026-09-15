"""Direct Ollama HTTP client for chat, model discovery, and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, get_settings


class OllamaError(RuntimeError):
    """Raised when Ollama cannot satisfy a request."""


@dataclass(slots=True)
class ChatResult:
    """Text generation result with model metadata."""

    content: str
    model: str
    used_fallback: bool = False


class OllamaService:
    """Small testable wrapper around Ollama's local HTTP API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")

    def list_models(self) -> list[str]:
        """Return installed model names."""

        timeout = httpx.Timeout(self.settings.ollama_connect_timeout_seconds)
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"Ollama model discovery failed: {exc}") from exc
        return [str(item.get("name", "")) for item in payload.get("models", []) if item]

    def health(self) -> tuple[bool, list[str]]:
        """Return reachability and installed models without raising."""

        try:
            return True, self.list_models()
        except OllamaError:
            return False, []

    def _chat_once(self, model: str, prompt: str, system: str | None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.settings.llm_temperature},
        }
        if self.settings.disable_thinking:
            payload["think"] = False

        timeout = httpx.Timeout(
            connect=self.settings.ollama_connect_timeout_seconds,
            read=self.settings.ollama_generation_timeout_seconds,
            write=30.0,
            pool=30.0,
        )
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        content = str(response.json().get("message", {}).get("content", "")).strip()
        if not content:
            raise OllamaError(f"{model} returned an empty response")
        return content

    def chat(self, prompt: str, system: str | None = None) -> ChatResult:
        """Generate text with primary then fallback model."""

        errors: list[str] = []
        models = [self.settings.primary_llm_model, self.settings.fallback_llm_model]
        for index, model in enumerate(dict.fromkeys(models)):
            try:
                return ChatResult(
                    content=self._chat_once(model=model, prompt=prompt, system=system),
                    model=model,
                    used_fallback=index > 0,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model}: {exc}")
        raise OllamaError("Ollama generation failed. " + " | ".join(errors))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for one or more texts."""

        if not texts:
            return []
        timeout = httpx.Timeout(
            connect=self.settings.ollama_connect_timeout_seconds,
            read=self.settings.ollama_generation_timeout_seconds,
            write=30.0,
            pool=30.0,
        )
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.settings.embedding_model, "input": texts},
                timeout=timeout,
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings", [])
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(f"Ollama embedding failed: {exc}") from exc
        if len(embeddings) != len(texts):
            raise OllamaError("Ollama returned an unexpected embedding count")
        return [[float(value) for value in vector] for vector in embeddings]
