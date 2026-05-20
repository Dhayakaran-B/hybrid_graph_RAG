GraphFusion AI
A Hybrid Semantic Graph RAG System built using Ollama, Qwen2.5, FAISS, NetworkX, spaCy, and Streamlit.
---
Overview
GraphFusion AI is a Hybrid Semantic Graph RAG (Retrieval-Augmented Generation) system that combines:
Vector-based semantic retrieval
Graph-based entity traversal
Local LLM inference using Ollama
Interactive knowledge graph visualization
Unlike traditional Vector RAG systems that rely only on embeddings, GraphFusion AI enriches retrieval using a semantic graph constructed from entity relationships extracted from text documents.
The project demonstrates how hybrid retrieval pipelines can improve contextual reasoning and retrieval quality by combining:
```text
Semantic Similarity + Graph Relationships
```
---
Core Architecture
```text
Document
   ↓
Chunking
   ↓
Entity Extraction (spaCy)
   ↓
Graph Creation (NetworkX)
   ↓
Embedding Generation (nomic-embed-text)
   ↓
FAISS Vector Indexing
   ↓
Hybrid Retrieval
   ↓
Qwen2.5 via Ollama
   ↓
Generated Answer
```
---
Features
Hybrid Graph + Vector RAG
Local LLM inference using Ollama
Semantic retrieval using FAISS
Entity-aware graph traversal
Interactive graph visualization
Streamlit frontend
Fully local execution
Explainable retrieval pipeline
---
Project Structure
```text
GraphFusion-AI/
│
├── data/
│   └── sample.txt
│
├── db/
│   ├── graph.pkl
│   ├── chunks.pkl
│   └── faiss\_index
│
├── build\_db.py
├── gRag.py
├── visualise.py
├── app.py
├── graph.html
└── requirements.txt
```
---
File Explanations
`build\_db.py`
The offline ingestion pipeline.
This script:
reads the source document
chunks the text
extracts entities using spaCy
builds a semantic graph using NetworkX
generates embeddings using `nomic-embed-text`
creates a FAISS vector index
saves the graph and vector database locally
Run this script whenever:
new documents are added
context changes
---
`gRag.py`
The core Hybrid Graph RAG engine.
This script:
accepts user queries
extracts entities from the query
traverses graph relationships
performs semantic vector retrieval
fuses graph + vector context
generates answers using Qwen2.5
This acts as the online reasoning pipeline.
---
`visualise.py`
Graph visualization engine.
This script:
loads the saved graph
converts it into an interactive PyVis network
generates `graph.html`
The graph visualization allows:
node traversal
interactive exploration
relationship visualization
---
`app.py`
Frontend layer built using Streamlit.
This script provides:
clean UI
question-answer interface
local inference interaction
frontend access to the RAG pipeline
It acts as the presentation layer for the project.
---
Technologies Used
Component	Technology
LLM Runtime	Ollama
Language Model	Qwen2.5
Embeddings	nomic-embed-text
Vector Search	FAISS
Graph Engine	NetworkX
NLP	spaCy
Frontend	Streamlit
Visualization	PyVis
---
Installation
1. Clone Repository
```bash
git clone <your-repo-link>
cd GraphFusion-AI
```
---
2. Create Virtual Environment
Windows
```bash
python -m venv venv
venv\\Scripts\\activate
```
Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```
---
3. Install Dependencies
```bash
pip install langchain
pip install langchain-community
pip install langchain-ollama
pip install langchain-text-splitters
pip install networkx
pip install spacy
pip install faiss-cpu
pip install numpy
pip install pandas
pip install pypdf
pip install tqdm
pip install pyvis
pip install matplotlib
pip install streamlit
```
---
4. Download spaCy Model
```bash
python -m spacy download en\_core\_web\_sm
```
---
5. Install Ollama
https://ollama.com
---
6. Pull Required Models
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```
---
Running the Project
Step 1 — Build Database
```bash
python build\_db.py
```
This creates:
```text
db/
├── graph.pkl
├── chunks.pkl
└── faiss\_index
```
---
Step 2 — Run CLI Graph RAG
```bash
python gRag.py
```
---
Step 3 — Generate Graph Visualization
```bash
python visualise.py
```
This generates:
```text
graph.html
```
---
Step 4 — Launch Frontend
```bash
streamlit run app.py
```
---
Example Queries
```text
What is CUDA?

How is IBM related to quantum computing?

What are transformers?

Who founded SpaceX?

How are neural networks used in autonomous driving?
```
---
Screenshots
Main Interface
(Add screenshot here)
---
Question Answering Demo
(Add screenshot here)
---
Graph Visualization
(Add screenshot here)
---
Hybrid Retrieval Pipeline
```text
User Query
   ↓
Entity Extraction (spaCy)
   ↓
Graph Traversal (NetworkX)

                +

Query Embedding
   ↓
FAISS Semantic Search

                ↓

Context Fusion
   ↓
Qwen2.5 via Ollama
   ↓
Generated Answer
```
---
Theory
Traditional Vector RAG systems rely purely on embedding similarity.
GraphFusion AI extends this by integrating:
entity-aware graph traversal
semantic relationships
hybrid retrieval
This improves:
contextual retrieval
relationship understanding
multi-hop reasoning
explainability
The project demonstrates a foundational implementation of a Hybrid Graph-Enhanced RAG Architecture.
---
Future Improvements
Neo4j integration
Typed semantic relationships
Multi-hop reasoning
Agentic retrieval
Streaming responses
PDF ingestion
Multi-document support
Cloud deployment
Relation extraction using LLMs
---
Author
Built as an experimental Hybrid Semantic Graph RAG system for exploring:
Graph RAG
Hybrid retrieval
Local LLM inference
Explainable AI systems
