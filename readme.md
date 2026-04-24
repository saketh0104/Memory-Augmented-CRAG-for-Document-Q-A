# Memory-Augmented Corrective RAG for Enterprise Document Question Answering

An intelligent Retrieval-Augmented Generation (RAG) system designed for enterprise-scale document understanding using adaptive retrieval, corrective evidence filtering, memory modules, and grounded answer generation.

---

## Overview

Large enterprise organizations manage vast collections of reports, filings, governance documents, policies, and operational records. Traditional search systems often fail to retrieve the most relevant evidence, while standard LLM systems may hallucinate unsupported answers.

This project introduces a **Memory-Augmented Corrective RAG (CRAG)** pipeline that improves document-grounded question answering through:

- Semantic retrieval using embeddings + vector database
- Intent-aware query routing
- Evidence quality filtering
- Query refinement for failed retrievals
- Episodic and evidence memory
- Grounded response generation with citations

The system is built for high-value enterprise use cases such as:

- Financial report analysis
- Board governance queries
- Policy understanding
- Internal knowledge retrieval
- Decision support systems

---

## Key Features

### Semantic Retrieval Engine
Uses Sentence Transformers to embed user queries and document chunks for dense semantic search.

### Corrective Retrieval (CRAG)
Low-quality evidence is filtered using an evidence critic. If retrieval quality is poor, the system automatically refines the query and retries.

### Memory-Augmented Reasoning
Supports:

- Episodic memory (past interactions)
- Validated evidence memory

to improve contextual continuity.

### Intent-Aware Pipeline
Different query types trigger different answering strategies:

- FACT_LOOKUP
- GLOBAL_SUMMARY
- PROCEDURAL
- EXPLORATORY

### Grounded Answer Generation
Answers are generated only from retrieved evidence and shown with chunk citations.

---

## System Architecture
<img width="15360" height="11520" alt="Architecture_8x" src="https://github.com/user-attachments/assets/2bf92aee-0ebc-48d1-a5d3-598bea2b99f3" />
---

## End-to-End Flow

```text
User Query
   ↓
Intent Router
   ↓
Semantic Retriever (ChromaDB)
   ↓
Evidence Critic
   ↓
Query Refinement (if needed)
   ↓
Memory Injection
   ↓
LLM Answer Generator
   ↓
Grounded Response + Citations
```

## Results:

<img width="1920" height="1080" alt="Screenshot 2026-03-27 141459" src="https://github.com/user-attachments/assets/68739cd7-4889-4918-b73f-03f762b05a0b" />

---
<img width="1920" height="1080" alt="Screenshot 2026-03-27 141557" src="https://github.com/user-attachments/assets/bd194c5e-fa35-43b4-b60a-a66abb21205e" />
