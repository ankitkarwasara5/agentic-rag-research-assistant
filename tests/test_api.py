from fastapi.testclient import TestClient

from app.main import create_app
from app.services.ollama import OllamaService


def test_root(settings):
    client = TestClient(create_app(settings))
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == "2.0.0"


def test_health_reports_degraded_without_ollama(settings, monkeypatch):
    monkeypatch.setattr(OllamaService, "health", lambda self: (False, []))
    client = TestClient(create_app(settings))
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["ollama_reachable"] is False


def test_health_reports_model_readiness(settings, monkeypatch):
    models = [settings.primary_llm_model, settings.embedding_model]
    monkeypatch.setattr(OllamaService, "health", lambda self: (True, models))
    client = TestClient(create_app(settings))
    payload = client.get("/health").json()
    assert payload["status"] == "healthy"
    assert payload["primary_model_available"] is True
    assert payload["embedding_model_available"] is True
