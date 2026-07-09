# Agentic RAG Research Assistant

A fully local Agentic RAG research assistant that runs on a MacBook without paid LLM APIs or API keys. It uses Ollama for local inference, LangGraph for the agent workflow, ChromaDB for vector storage, FastAPI for the backend, and Streamlit for the UI.

The defaults are tuned for an Apple Silicon MacBook M3 with 16 GB RAM. Ollama uses Metal acceleration on macOS when available, and the project keeps the default context and generation sizes conservative for `qwen3:8b`.

## Architecture

```text
Streamlit UI
  |
  v
FastAPI POST /research
  |
  v
LangGraph workflow
  |-- Query Understanding Agent
  |-- Research Planning Agent
  |-- Web Search Agent using DuckDuckGo
  |-- Document Ingestion Agent for PDF/TXT/MD
  |-- Retrieval Agent using ChromaDB + Ollama embeddings
  |-- Summarization Agent
  |-- Citation Validation Agent
  |-- Final Report Agent
  |
  v
Citation-backed Markdown report
```

## Features

- Local-only LLM inference with Ollama.
- Default model: `qwen3:8b`.
- Fallback model: `llama3.2:3b`.
- Local embedding model: `nomic-embed-text`.
- Free DuckDuckGo web search.
- PDF, TXT, and Markdown document ingestion.
- Persistent ChromaDB vector database.
- Agentic workflow for query refinement, planning, search, retrieval, summarization, validation, and report writing.
- Citation labels like `[Source 1]`.
- Confidence score and warnings when search, ingestion, embeddings, or Ollama calls fail.
- Downloadable Markdown report.

## Setup

Use Python 3.11 or newer.

```bash
cd agentic-rag-research-assistant
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama on macOS:

```bash
brew install ollama
brew services start ollama
```

Pull the required local models:

```bash
ollama pull qwen3:8b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Optional configuration:

```bash
cp .env.example .env
```

The M3 defaults in `.env.example` are:

```bash
RAG_OLLAMA_NUM_CTX=4096
RAG_OLLAMA_NUM_PREDICT=1400
RAG_OLLAMA_NUM_THREAD=8
RAG_OLLAMA_NUM_GPU=1
RAG_OLLAMA_KEEP_ALIVE=10m
```

## Run

Backend:

```bash
uvicorn app.main:app --reload
```

Frontend:

```bash
streamlit run frontend/streamlit_app.py
```

Or run both:

```bash
chmod +x run.sh
./run.sh
```

Open the Streamlit URL, enter a research query, optionally upload PDF/TXT/MD files, and click `Run Research`.

## API

Multipart request with optional files:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -F "query=Compare RAG and Agentic RAG" \
  -F "files=@data/samples/agentic_rag_notes.md"
```

JSON request without files:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query":"Research the latest trends in Agentic AI"}'
```

Response fields:

- `query`
- `research_plan`
- `retrieved_context`
- `final_report`
- `citations`
- `confidence_score`
- `refined_questions`
- `warnings`

## Example Queries

- "Research the latest trends in Agentic AI"
- "Compare RAG and Agentic RAG"
- "Summarize AI regulation developments"

## Tests

```bash
pytest
```

The tests avoid live Ollama and web calls. Runtime research requires Ollama to be running and the three models to be pulled.

## Failure Handling

- If web search fails, uploaded documents can still be ingested and retrieved.
- If no files are uploaded, the workflow uses web sources.
- If ChromaDB or embeddings fail, the retrieval agent falls back to local lexical retrieval and adds a warning.
- If the primary LLM fails, the wrapper tries `llama3.2:3b`.
- If both LLMs fail, the app returns a deterministic source-backed fallback report and warnings.

## Resume Bullets

- Built a fully local Agentic RAG Research Assistant using Qwen3, LangGraph, ChromaDB, and Ollama without paid LLM APIs.
- Designed a multi-agent research workflow for query planning, web search, document ingestion, retrieval, summarization, validation, and citation-backed report generation.
- Implemented vector search using local embeddings and ChromaDB to retrieve relevant context from PDFs, text files, and web sources.
- Added hallucination-reduction logic through citation validation, confidence scoring, and source-backed final answers.
- Deployed the system with FastAPI and Streamlit for real-time research automation and downloadable Markdown reports.
