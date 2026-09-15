# Architecture

The application deliberately separates orchestration, inference, retrieval,
and evaluation so each piece can be tested independently.

```text
Streamlit
   |
FastAPI
   |
LangGraph
   |-- Query decomposition
   |-- Research planning
   |-- Web + document collection
   |-- Semantic retrieval (Ollama embeddings + Chroma)
   |      \-- deterministic lexical fallback
   |-- Evidence synthesis
   |-- Citation-backed report drafting
   |-- Deterministic evaluation
   |      \-- one revision pass when quality is below threshold
   |
Markdown report + citations + evaluation metrics
```

## Why direct Ollama and Chroma APIs?

The project avoids unnecessary model/vector wrappers. The Ollama adapter owns
chat, model discovery, and embeddings. Chroma receives explicit embeddings and
metadata. This keeps failures observable and makes fallbacks straightforward.

## Evaluation

The evaluator reports five transparent heuristics:

- citation validity: whether cited labels exist in retrieved evidence;
- citation coverage: how many substantive paragraphs include citations;
- source diversity: number of independent retrieved sources;
- retrieval quality: average normalized retrieval score;
- groundedness proxy: lexical overlap between cited paragraphs and their
  evidence chunks.

The groundedness metric is intentionally called a proxy. It is useful for
regression testing and triage, not a substitute for factual verification.
