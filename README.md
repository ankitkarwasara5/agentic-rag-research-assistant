# Agentic RAG Research Assistant

A local-first research system that plans a research task, collects web and
uploaded-document evidence, retrieves relevant chunks, writes a citation-backed
report, and evaluates the result before returning it.

It is designed as a portfolio-quality example of **agentic AI, RAG, evaluation,
local LLM inference, retrieval engineering, and production-style Python APIs**.

## What makes this more than a basic RAG demo?

- **LangGraph orchestration** with an explicit evaluation/revision loop.
- **Local Ollama inference** with primary/fallback models and real health checks.
- **Direct Ollama embeddings + ChromaDB** without unnecessary wrapper layers.
- **Hybrid retrieval**: semantic retrieval with deterministic lexical fallback.
- **Web + document evidence** from PDF, TXT, and Markdown files.
- **Citation-constrained reporting**: generated reports may use only retrieved
  source labels.
- **Transparent evaluation** for citation validity, citation coverage, source
  diversity, retrieval quality, and a groundedness proxy.
- **Failure-aware behavior**: search, embeddings, and generation can fail without
  silently pretending the run succeeded normally.
- **FastAPI + Streamlit + Docker + CI + tests**.

## Architecture

```text
Streamlit UI
    |
    v
FastAPI  POST /research
    |
    v
LangGraph workflow
    |
    |-- Query decomposition
    |-- Research planning
    |-- Web + document collection
    |-- Retrieval
    |     |-- Ollama embeddings + ChromaDB
    |     \-- lexical fallback
    |-- Evidence synthesis
    |-- Citation-backed report
    |-- Deterministic evaluator
    |     \-- optional one-pass revision
    |
    v
Report + citations + metrics + diagnostics
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

## Tech stack

- Python 3.11+
- FastAPI
- LangGraph
- Ollama (`qwen3:8b` by default)
- ChromaDB
- `nomic-embed-text`
- DuckDuckGo search through `ddgs`
- PyMuPDF for PDF extraction
- Streamlit
- Pytest + Ruff
- Docker / Docker Compose

## Quick start

### 1. Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 2. Prepare Ollama

```bash
ollama serve
```

In another terminal:

```bash
ollama pull qwen3:8b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Configure

```bash
cp .env.example .env
```

### 4. Run the API

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check readiness:

```bash
curl http://127.0.0.1:8000/health | python -m json.tool
```

### 5. Run the UI

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://127.0.0.1:8501`.

## API example

Web-only research:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare standard RAG and agentic RAG"}'
```

Research with a local document:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -F "query=Compare standard RAG and agentic RAG" \
  -F "files=@data/samples/agentic_rag_notes.md"
```

The response includes the research plan, refined questions, retrieved evidence,
final report, citations, evaluation metrics, warnings, model used, retrieval
mode, and revision count.

## Evaluation layer

The project intentionally exposes quality diagnostics rather than hiding them
behind one score:

| Metric | What it checks |
| --- | --- |
| Citation validity | Cited labels exist in retrieved evidence |
| Citation coverage | Substantive paragraphs include citations |
| Source diversity | Evidence comes from multiple independent sources |
| Retrieval quality | Average normalized retrieval score |
| Groundedness proxy | Lexical overlap between cited claims and evidence |

These are **engineering heuristics**, not claims of factual correctness. The
proxy naming is deliberate so the UI does not overstate what an automated
metric can prove.

When the overall score is below the configured threshold, LangGraph can run one
report-revision pass using the same allowed evidence.

## Tests and quality

```bash
python -m pytest
ruff check .
ruff format --check .
```

Tests are deterministic and do not require live Ollama, Chroma, or web access.
The live dependencies are exercised during manual testing.

## Docker

Keep Ollama running on your Mac, then:

```bash
docker compose up --build
```

- UI: `http://127.0.0.1:8501`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

Docker Desktop reaches the host Ollama service through
`host.docker.internal:11434`.

## Project structure

```text
.
├── app/
│   ├── agents/
│   ├── evaluation/
│   ├── services/
│   ├── utils/
│   ├── config.py
│   ├── graph.py
│   ├── main.py
│   └── schemas.py
├── frontend/
├── tests/
├── docs/
├── data/
├── reports/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Design choices

### Direct Ollama API

The model adapter uses Ollama's `/api/chat`, `/api/tags`, and `/api/embed`
endpoints directly. This keeps inference and embedding failures observable and
avoids coupling the project to an extra LLM wrapper.

### Deterministic fallback retrieval

If Chroma or Ollama embeddings are unavailable, retrieval falls back to lexical
term overlap and records a warning. This provides graceful degradation without
claiming semantic retrieval succeeded.

### Evaluation before completion

The graph evaluates its own report against the evidence contract. Low-scoring
reports can be revised once, creating a visible long-horizon loop rather than a
single prompt-response pipeline.

## Security / limitations

- Uploaded files are limited in size and restricted to PDF/TXT/Markdown.
- Web loading rejects obvious localhost/private-IP URLs.
- Generated reports can still be wrong even when citations are valid.
- The groundedness metric is lexical and should be treated as a regression
  signal, not a factuality oracle.
- DuckDuckGo results and websites can be unavailable or rate limited.

## Manual testing

See [`MANUAL_TESTING.md`](MANUAL_TESTING.md).

## License

MIT
