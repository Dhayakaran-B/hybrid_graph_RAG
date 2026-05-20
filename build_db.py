import os
import pickle
import networkx as nx
import spacy
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# =========================
# CONFIG
# =========================
DATA_PATH = "data/sample.txt"
DB_FOLDER = "db"

os.makedirs(DB_FOLDER, exist_ok=True)

# =========================
# LOAD SPACY
# =========================
print("\nLoading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# =========================
# LOAD DOCUMENT
# =========================
print("Loading document...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# CHUNKING
# =========================
print("Chunking document...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")

# =========================
# GRAPH BUILDING
# =========================
print("Building graph...")

G = nx.Graph()

for chunk_id, chunk in enumerate(chunks):

    doc = nlp(chunk)

    entities = [ent.text.strip() for ent in doc.ents]

    print(f"\nChunk {chunk_id} entities: {entities}")

    # Add nodes
    for entity in entities:
        G.add_node(entity)

    # Add edges between entities in same chunk
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            G.add_edge(
                entities[i],
                entities[j],
                chunk_id=chunk_id
            )

print(f"\nGraph nodes: {len(G.nodes)}")
print(f"Graph edges: {len(G.edges)}")

# =========================
# LOAD EMBEDDING MODEL
# =========================
print("\nLoading embedding model...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# =========================
# CREATE EMBEDDINGS
# =========================
print("Creating embeddings...")

try:
    chunk_embeddings = embeddings.embed_documents(chunks)
    print("Embeddings created successfully!")

except Exception as e:
    print("\nERROR while creating embeddings:")
    print(e)
    exit()

# =========================
# CREATE FAISS INDEX
# =========================
print("\nCreating FAISS index...")

embedding_dim = len(chunk_embeddings[0])

index = faiss.IndexFlatL2(embedding_dim)

index.add(np.array(chunk_embeddings, dtype=np.float32))

print("FAISS index created!")

# =========================
# SAVE DATABASE
# =========================
print("\nSaving database files...")

try:

    with open(os.path.join(DB_FOLDER, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)

    with open(os.path.join(DB_FOLDER, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    faiss.write_index(
        index,
        os.path.join(DB_FOLDER, "faiss_index")
    )

    print("\nDatabase successfully built!")

except Exception as e:
    print("\nERROR while saving database:")
    print(e)