# Agentic RAG notes

Standard retrieval-augmented generation typically retrieves evidence for a
query and provides that evidence to a language model before generation.

Agentic RAG adds workflow decisions around retrieval. An agent may decompose a
question, plan research steps, use multiple sources, inspect retrieval quality,
and revise the answer when evidence coverage is weak.

A useful evaluation should separate retrieval quality, citation validity,
citation coverage, source diversity, and groundedness rather than collapsing
all quality into a single opaque score.
