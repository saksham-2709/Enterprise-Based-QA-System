# Enterprise Policy Intelligence System (RAG)

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed for policy search, factual Q&A, and compliance decision-making across organizational documents.

Unlike basic RAG chatbots, this system prioritizes correctness, explainability, and safety, making it suitable for HR, legal, and internal policy use cases.

---

## Key Features

- Semantic document search using FAISS
- Question Answering (QA) with concise, grounded answers
- Decision reasoning with ALLOWED / NOT ALLOWED / UNCLEAR outcomes
- Numeric policy reasoning (e.g. leave limits)
- Automatic mode detection (Search / QA / Decision)
- Multi-organization support with isolated vector indexes
- Chunk reconstruction to handle fragmented policy rules
- Safe refusal when evidence is insufficient
- Automated evaluation framework using real-world policies

---

## Why This Project Is Different

Most RAG projects:
- Behave like chatbots
- Hallucinate answers
- Mix conflicting documents
- Have no testing or evaluation

This system:
- Separates search, QA, and decision logic
- Uses deterministic, rule-based reasoning
- Avoids hallucination by design
- Handles real enterprise policy conflicts
- Includes automated testing for correctness

---

## Architecture Overview

Document Ingestion
  → Chunking & Metadata
  → Embeddings (Sentence Transformers)
  → FAISS Vector Index (per organization)
  → Retriever
  → Mode Router
      - Search
      - QA
      - Decision
  → Confidence Scoring & Evaluation

---

## Project Structure

RAG/
├── ingestion/          # document loaders, chunking, embeddings
├── retrieval/          # FAISS retrieval logic
├── modes/              # search, qa, decision implementations
├── orchestration/      # pipeline, mode guessing, context handling
├── evaluation/         # automated evaluation framework
├── data/
│   ├── gitlab/         # GitLab policy documents
│   └── generic_company/
├── indexes/            # FAISS indexes per organization
├── main.py             # CLI entry point
└── faiss_index.py

---


## Evaluation & Testing

The project includes an automated evaluation harness that validates:

- Correct mode selection
- Correct policy reasoning
- Numeric decision accuracy
- Evidence grounding
- Safe failure behavior

Run evaluation:
  python evaluation/run_evaluation.py

All tests pass on real GitLab Handbook policy data.

---

## Design Principles

- LLMs are optional
  Core logic works without OpenAI or external APIs
- No hallucination
  Answers are strictly grounded in retrieved evidence
- Explainability
  Every decision includes reasons and source evidence
- Enterprise safety
  Conflicting policies are isolated via separate indexes

---

## Tech Stack

- Python
- FAISS
- Sentence Transformers
- Retrieval-Augmented Generation (RAG)
- Rule-based reasoning
- Automated evaluation

---

