# System Design

## Overview
A modular legal intelligence system that answers precedent-driven judiciary questions using RAG and an agentic workflow.

Components:
- Data: compact corpus with metadata (title, year, court, tags).
- Embedder: `sentence-transformers` MiniLM.
- Index: In-memory NumPy matrix with normalized vectors (cosine similarity).  // FAISS removed for portability
- Retriever: similarity blended with recency, court weight, and issue overlap; applies small feedback-based boosts.
- LLM: NVIDIA Inference API via OpenAI client, streaming-safe.
- Agent: multi-step orchestration (issues → retrieval → plan → grounding → arguments → verdict).
- Feedback: simple JSON-driven style adapter and per-document ranking nudges.
- UI: Streamlit app with temperature and max-token controls passed to drafting.

## Flow
1. Ingest corpus and build embeddings.
2. Query formed from case title + facts (issues hints optional).
3. Retrieve top precedents using composite scoring.
4. Generate arguments and verdict using LLM with explicit proportionality checklist and strict sectioning.
5. Save outputs and allow optional feedback to adapt style and ranking.

## Agent Steps
- Issue Spotting: keyword extractor across privacy, proportionality, biometrics, safeguards, legality, religion, expression, equality, trade, assembly, internet, localization, surveillance.
- Planning: legality → suitability → necessity → balancing/safeguards → outcome (+ branches per detected issues).
- Grounding: link claims to sources (titles/years) for traceability.
- Drafting: petitioner and state sides with citations; verdict with confidence.

## Extensibility
- Swap NumPy retriever for FAISS/Pinecone/Weaviate if needed.
- Add multi-hop retrieval or case graph traversal.
- Introduce evaluator to score factual grounding and policy compliance.
- Support additional jurisdictions by enriching corpus and tag schema.
