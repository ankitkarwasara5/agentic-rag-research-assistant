#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

mkdir -p data/uploads data/chroma_db reports

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed or not on PATH. Install it before running research."
fi

uvicorn app.main:app --reload --host 127.0.0.1 --port "${RAG_API_PORT:-8000}" &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 2
streamlit run frontend/streamlit_app.py --server.port "${RAG_STREAMLIT_PORT:-8501}"
