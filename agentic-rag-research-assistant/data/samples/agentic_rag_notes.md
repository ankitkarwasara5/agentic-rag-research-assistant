# Agentic RAG Notes

Agentic RAG extends retrieval-augmented generation by adding planning,
tool use, iterative retrieval, and validation steps. A typical workflow
clarifies the user query, creates subquestions, retrieves evidence, checks
whether claims are supported, and produces a citation-backed answer.

Local-first systems can run with Ollama for LLM inference, a local embedding
model for vector search, and ChromaDB for persistent document retrieval.

