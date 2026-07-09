"""Local Ollama LLM wrapper with primary/fallback model support."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.utils.text_utils import strip_thinking

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - exercised only without dependencies
    ChatOllama = None  # type: ignore[assignment]


class LLMUnavailableError(RuntimeError):
    """Raised when no configured Ollama model can generate a response."""


@dataclass
class LLMResponse:
    """Generated text with model metadata."""

    content: str
    model: str
    used_fallback_model: bool = False


class LocalLLM:
    """Thin wrapper around LangChain's ChatOllama integration."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _build_model(self, model_name: str):
        if ChatOllama is None:
            raise LLMUnavailableError(
                "langchain-ollama is not installed. Run `pip install -r requirements.txt`."
            )
        return ChatOllama(
            model=model_name,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.llm_temperature,
            num_ctx=self.settings.ollama_num_ctx,
            num_predict=self.settings.ollama_num_predict,
            num_thread=self.settings.ollama_num_thread,
            num_gpu=self.settings.ollama_num_gpu,
            keep_alive=self.settings.ollama_keep_alive,
        )

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate text, trying qwen3:8b before the fallback model."""

        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))

        errors: list[str] = []
        for idx, model_name in enumerate(
            [self.settings.primary_llm_model, self.settings.fallback_llm_model]
        ):
            try:
                model = self._build_model(model_name)
                response = model.invoke(messages)
                content = strip_thinking(getattr(response, "content", str(response)))
                if content:
                    return LLMResponse(
                        content=content,
                        model=model_name,
                        used_fallback_model=idx > 0,
                    )
                errors.append(f"{model_name}: empty model response")
            except Exception as exc:  # noqa: BLE001 - keep API resilient
                errors.append(f"{model_name}: {exc}")

        detail = " | ".join(errors) if errors else "unknown Ollama error"
        raise LLMUnavailableError(
            "Ollama generation failed. Make sure Ollama is running and the models "
            f"are pulled. Details: {detail}"
        )

    def generate_or_fallback(
        self,
        prompt: str,
        fallback: str,
        warnings: list[str],
        system: str | None = None,
    ) -> str:
        """Generate text and append a warning if deterministic fallback is used."""

        try:
            return self.generate(prompt=prompt, system=system).content
        except LLMUnavailableError as exc:
            warnings.append(str(exc))
            return fallback


def get_local_llm() -> LocalLLM:
    """Factory used by graph nodes."""

    return LocalLLM()

