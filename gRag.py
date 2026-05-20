import pickle
import faiss
import networkx as nx
import numpy as np
import spacy

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# =========================
# LOAD DATABASE
# =========================
DB_FOLDER = "db"

print("\nLoading graph...")

with open(f"{DB_FOLDER}/graph.pkl", "rb") as f:
    G = pickle.load(f)

with open(f"{DB_FOLDER}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index(f"{DB_FOLDER}/faiss_index")

print("Database loaded successfully!")

# =========================
# LOAD SPACY
# =========================
print("Loading spaCy...")

nlp = spacy.load("en_core_web_sm")

# =========================
# LOAD EMBEDDINGS
# =========================
print("Loading embeddings...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# =========================
# LOAD QWEN
# =========================
print("Loading Qwen model...")

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

print("Qwen loaded!")

# =========================
# QUERY LOOP
# =========================
while True:

    query = input("\nAsk something (or type exit): ")

    if query.lower() == "exit":
        break

    # =========================
    # ENTITY EXTRACTION
    # =========================
    doc = nlp(query)

    query_entities = [ent.text.strip() for ent in doc.ents]

    print(f"\nDetected entities: {query_entities}")

    # =========================
    # GRAPH SEARCH
    # =========================
    graph_context = []

    for entity in query_entities:

        if entity in G:

            neighbors = list(G.neighbors(entity))

            graph_context.extend(neighbors)

    graph_context = list(set(graph_context))

    print(f"\nGraph Context: {graph_context}")

    # =========================
    # VECTOR SEARCH
    # =========================
    print("\nSearching vector database...")

    query_embedding = embeddings.embed_query(query)

    D, I = index.search(
        np.array([query_embedding], dtype=np.float32),
        k=3
    )

    retrieved_chunks = [chunks[i] for i in I[0]]

    print("\nRetrieved Chunks:\n")

    for chunk in retrieved_chunks:
        print(chunk)
        print("-" * 50)

    # =========================
    # FINAL CONTEXT
    # =========================
    final_context = "\n".join(retrieved_chunks)

    final_context += "\n\nGraph Related Entities:\n"

    final_context += ", ".join(graph_context)

    # =========================
    # PROMPT
    # =========================
    prompt = f"""
You are a Semantic Graph RAG assistant.

Use ONLY the provided context to answer.

Context:
{final_context}

Question:
{query}

Answer:
"""

    # =========================
    # GENERATE ANSWER
    # =========================
    print("\nGenerating answer...\n")

    try:

        response = llm.invoke(prompt)

        print(response.content)

    except Exception as e:

        print("\nERROR while generating response:")
        print(e)