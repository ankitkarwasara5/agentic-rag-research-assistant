# Manual testing

## 1. Prepare Ollama

```bash
ollama serve
ollama pull qwen3:8b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Verify:

```bash
curl http://127.0.0.1:11434/api/tags
```

## 2. Run backend

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Check:

```bash
curl http://127.0.0.1:8000/health | python -m json.tool
```

A fully ready local environment should report `status: healthy` and show the
configured LLM and embedding models as available.

## 3. Run frontend

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://127.0.0.1:8501`.

## 4. Research tests

Run these separately:

1. Web-only query: `Compare standard RAG and agentic RAG.`
2. Document-only resilience: temporarily disable internet access, upload
   `data/samples/agentic_rag_notes.md`, and run the same query.
3. Mixed evidence: upload the sample and keep web search enabled.
4. Unsupported upload: verify an image or executable is rejected.
5. Ollama stopped: verify the API returns warnings and source-backed fallbacks
   rather than fabricated model content.

Inspect the diagnostics panel for retrieval mode, evaluation metrics, revision
count, and runtime warnings.

## 5. Docker

On Docker Desktop for macOS, keep Ollama running on the host and run:

```bash
docker compose up --build
```

Then open `http://127.0.0.1:8501`.
