Memory-Augmented Corrective Agentic RAG for Reliable Context-Aware Document Question Answering

A research-oriented implementation of a Memory-Augmented Corrective Retrieval-Augmented Generation (RAG) system designed for reliable enterprise document analysis and decision support.

This system extends traditional RAG by integrating:

Intent-aware query routing

Corrective Retrieval (CRAG)

Persistent episodic and evidence memory

Faithfulness validation

Session-based interaction management

🚀 Overview

Organizations rely on large collections of unstructured documents such as:

Policies

Financial reports

Governance disclosures

Operational manuals

Traditional RAG systems often retrieve partially relevant context and may generate unsupported responses.

This project introduces:

Adaptive retrieval thresholding

Evidence validation before generation

Corrective re-retrieval via query refinement

Memory-augmented response consistency

Citation-supported grounded answers

The system returns:

Final Answer

Citations (source file + chunk_id)

Intent classification

Confidence-based gating

🧠 Core Architecture

The system consists of:

Document Ingestion Pipeline

Embedding & Vector Storage (ChromaDB)

Agentic Retrieval

Evidence Quality Critic (CRAG)

Query Refinement

Memory-Augmented Generator

Faithfulness Validation

Persistent Session Management

📂 Project Structure
memo_rag/
│
├── ingestion/        # Document loading & preprocessing
├── embeddings/       # Embedding model wrapper
├── vectorstore/      # ChromaDB integration
├── agents/           # Retrieval & corrective agents
├── rag/              # End-to-end pipeline
├── memory/           # Episodic & evidence memory
├── evaluation/       # Retrieval & faithfulness metrics
├── templates/        # UI
├── static/           # CSS & JS
├── data/             # Raw & processed documents
└── scripts/          # Utility scripts
🔄 Retrieval Workflow

User submits query

Intent Router classifies query type

Adaptive threshold configuration applied

Semantic retrieval from vector database

CRAG evaluates evidence quality

If needed → Query refinement and re-retrieval

Generator produces grounded response

Faithfulness validator checks citation support

Memory updated (if high confidence)

Response returned with citations

📊 Evaluation Design (In Progress)

Planned evaluation includes:

Recall@k

MRR

Faithfulness rate

Intent classification accuracy

Abstention accuracy (unanswerable detection)

Ablation: Vanilla RAG vs Corrective Memory-Augmented RAG

Dataset source: Public enterprise documents (e.g., SEC filings)

🖥️ Web Interface Features

Document upload

Session-based chat

Rename / delete sessions

Citation display

Persistent session history

⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/Memory-Augmented-CRAG-for-Document-Q-A.git
cd Memory-Augmented-CRAG-for-Document-Q-A

Create virtual environment:

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py
🧪 Current Status

This is an active research implementation.
Core components implemented:

Agentic CRAG pipeline

Persistent vector store

Memory modules

Session management

Ongoing work:

Evaluation framework completion

Ablation experiments

Confidence score exposure in UI

Dataset benchmarking

📌 Research Focus

This project investigates:

Limitations of traditional RAG systems

Adaptive retrieval strategies

Memory-augmented response generation

Reliable citation-grounded document QA

📜 License

For academic and research use.
