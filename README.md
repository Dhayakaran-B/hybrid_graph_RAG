# GraphFusion AI

Hybrid Semantic Graph RAG built using Ollama, Qwen2.5, FAISS, NetworkX, spaCy, and Streamlit.

---

## Overview

GraphFusion AI combines:

- Vector-based semantic retrieval
- Graph-based entity traversal
- Local LLM inference
- Interactive graph visualization

Instead of relying only on embeddings, the system also retrieves connected entities from a semantic graph to improve contextual reasoning.

---

## Architecture

```text
Document
   ↓
Chunking
   ↓
Entity Extraction
   ↓
Graph Creation
   ↓
Embedding Generation
   ↓
FAISS Indexing
   ↓
Hybrid Retrieval
   ↓
Qwen2.5 via Ollama
   ↓
Generated Answer
```

---

## Features

- Hybrid Graph + Vector RAG
- Local LLM inference with Ollama
- Semantic retrieval using FAISS
- Graph traversal using NetworkX
- Streamlit frontend
- Interactive graph visualization
- Fully local execution

---

## Project Structure

```text
GraphFusion-AI/
│
├── data/
├── db/
├── build_db.py
├── gRag.py
├── visualise.py
├── app.py
└── requirements.txt
```

---

## File Descriptions

### build_db.py

Offline ingestion pipeline.

Responsible for:
- chunking documents
- extracting entities using spaCy
- building graph using NetworkX
- generating embeddings
- creating FAISS index
- saving database files

Run this whenever the dataset changes.

---

### gRag.py

Core Hybrid Graph RAG engine.

Responsible for:
- query processing
- graph traversal
- vector retrieval
- context fusion
- answer generation using Qwen2.5

---

### visualise.py

Graph visualization engine.

Responsible for:
- loading graph.pkl
- converting graph into PyVis network
- generating graph.html

---

### app.py

Frontend built using Streamlit.

Provides:
- question-answer interface
- local inference interaction
- clean UI for the RAG pipeline

---

## Technologies Used

| Component | Technology |
|---|---|
| LLM Runtime | Ollama |
| Language Model | Qwen2.5 |
| Embeddings | nomic-embed-text |
| Vector Search | FAISS |
| Graph Engine | NetworkX |
| NLP | spaCy |
| Frontend | Streamlit |
| Visualization | PyVis |

---

## Installation

### Clone Repository

```bash
git clone <your-repo-link>
cd GraphFusion-AI
```

---

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install langchain
pip install langchain-community
pip install langchain-ollama
pip install langchain-text-splitters
pip install networkx
pip install spacy
pip install faiss-cpu
pip install numpy
pip install pyvis
pip install streamlit
```

---

### Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

### Install Ollama

https://ollama.com

---

### Pull Required Models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

---

## Running the Project

### Build Database

```bash
python build_db.py
```

---

### Run Graph RAG Engine

```bash
python gRag.py
```

---

### Generate Graph Visualization

```bash
python visualise.py
```

---

### Launch Frontend

```bash
streamlit run app.py
```

---

## Example Queries

```text
What is CUDA?

How is IBM related to quantum computing?

What are transformers?

Who founded SpaceX?
```

---

## Screenshots

### Main Interface

(Add screenshot here)

---

### Question Answering Demo

(Add screenshot here)

---

### Graph Visualization

(Add screenshot here)

---

## Hybrid Retrieval Pipeline

```text
User Query
   ↓
Entity Extraction
   ↓
Graph Traversal

            +

Query Embedding
   ↓
FAISS Search

            ↓

Context Fusion
   ↓
Qwen2.5
   ↓
Generated Answer
```

---

## Theory

Traditional Vector RAG systems rely only on embedding similarity.

GraphFusion AI extends retrieval using:
- semantic graph traversal
- entity-aware retrieval
- hybrid context fusion

This improves:
- contextual understanding
- retrieval quality
- explainability
- relational reasoning

---

## Future Improvements

- Neo4j integration
- Multi-hop reasoning
- Streaming responses
- PDF ingestion
- Multi-document support
- Agentic retrieval

---

## Author

Experimental Hybrid Semantic Graph RAG system for exploring:
- Graph RAG
- Hybrid retrieval
- Local LLM inference
- Explainable AI systems
