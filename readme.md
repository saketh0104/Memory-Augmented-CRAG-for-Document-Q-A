memo_rag/
│
├── app.py                         # Flask entry point
├── config.py                      # Global config (paths, model names)
├── requirements.txt               # All dependencies
├── .env                           # API keys (ignored in git)
├── README.md                      # Project overview
│
├── data/
│   ├── raw/                       # Uploaded documents (PDF, TXT)
│   ├── processed/                # Cleaned & chunked text
│   ├── metadata/                 # Extracted metadata (JSON)
│   └── samples/                  # Sample docs for testing
│
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                  # PDF / TXT / Web loaders
│   ├── cleaner.py                 # Text cleaning & normalization
│   ├── chunker.py                 # Semantic chunking
│   └── metadata_extractor.py      # Title, sections, timestamps
│
├── embeddings/
│   ├── __init__.py
│   ├── embedder.py                # Embedding model wrapper
│   └── embedding_utils.py         # Batch + caching helpers
│
├── vectorstore/
│   ├── __init__.py
│   ├── chroma_store.py            # ChromaDB init & CRUD
│   └── faiss_index.py             # FAISS backend helpers
│
├── agents/
│   ├── __init__.py
│   ├── query_understanding.py     # Query Understanding Agent
│   ├── retriever.py               # Retriever Agent
│   ├── evidence_critic.py         # CRAG (Evidence Quality Critic)
│   ├── query_refiner.py           # Query Refinement Agent
│   └── memory_manager.py          # Episodic & semantic memory
│
├── rag/
│   ├── __init__.py
│   ├── pipeline.py                # End-to-end RAG flow
│   ├── generator.py               # Memory-Augmented Generator
│   └── faithfulness.py            # Faithfulness Validator
│
├── memory/
│   ├── __init__.py
│   ├── episodic_memory.py         # Past Q&A storage
│   ├── semantic_memory.py         # Distilled knowledge
│   └── failure_memory.py          # Retrieval failure logs
│
├── web/
│   ├── __init__.py
│   ├── routes.py                  # Flask routes
│   ├── forms.py                   # Upload/query forms
│   └── utils.py                   # Web helpers
│
├── templates/
│   ├── index.html                 # Upload + query UI
│   ├── results.html               # Answer + citations
│   └── layout.html                # Base template
│
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
│
├── evaluation/
│   ├── __init__.py
│   ├── retrieval_metrics.py       # Recall@k, MRR
│   ├── faithfulness_metrics.py    # Hallucination checks
│   └── ablation_tests.py          # Vanilla RAG vs MEMO-RAG
│
├── logs/
│   ├── app.log
│   ├── retrieval.log
│   └── errors.log
│
└── scripts/
    ├── ingest_documents.py        # CLI ingestion
    ├── reset_vectorstore.py       # Cleanup utilities
    └── demo_queries.py            # Sample queries
