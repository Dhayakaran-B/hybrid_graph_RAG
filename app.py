import streamlit as st
import pickle
import faiss
import numpy as np
import spacy
import networkx as nx

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Semantic Graph RAG",
    layout="centered"
)

st.title("Hybrid Semantic Graph RAG")

st.write("Ask questions based on the uploaded knowledge base.")

# =========================
# LOAD DATABASE
# =========================
DB_FOLDER = "db"

@st.cache_resource
def load_resources():

    # Load Graph
    with open(f"{DB_FOLDER}/graph.pkl", "rb") as f:
        G = pickle.load(f)

    # Load Chunks
    with open(f"{DB_FOLDER}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load FAISS
    index = faiss.read_index(f"{DB_FOLDER}/faiss_index")

    # Load spaCy
    nlp = spacy.load("en_core_web_sm")

    # Load Embedding Model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Load Qwen
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    return G, chunks, index, nlp, embeddings, llm

G, chunks, index, nlp, embeddings, llm = load_resources()

# =========================
# USER QUERY
# =========================
query = st.text_input(
    "Enter your question"
)

# =========================
# QUERY PROCESSING
# =========================
if query:

    # -------------------------
    # ENTITY EXTRACTION
    # -------------------------
    doc = nlp(query)

    query_entities = [
        ent.text.strip()
        for ent in doc.ents
    ]

    # -------------------------
    # GRAPH RETRIEVAL
    # -------------------------
    graph_context = []

    for entity in query_entities:

        if entity in G:

            neighbors = list(G.neighbors(entity))

            graph_context.extend(neighbors)

    graph_context = list(set(graph_context))

    # -------------------------
    # VECTOR SEARCH
    # -------------------------
    query_embedding = embeddings.embed_query(query)

    D, I = index.search(
        np.array([query_embedding], dtype=np.float32),
        k=3
    )

    retrieved_chunks = [
        chunks[i]
        for i in I[0]
    ]

    # -------------------------
    # FINAL CONTEXT
    # -------------------------
    final_context = "\n".join(retrieved_chunks)

    final_context += "\n\nGraph Related Entities:\n"

    final_context += ", ".join(graph_context)

    # -------------------------
    # PROMPT
    # -------------------------
    prompt = f"""
You are a Semantic Graph RAG assistant.

Use ONLY the provided context to answer.

Context:
{final_context}

Question:
{query}

Answer:
"""

    # -------------------------
    # GENERATE RESPONSE
    # -------------------------
    with st.spinner("Generating answer..."):

        response = llm.invoke(prompt)

    # -------------------------
    # DISPLAY ANSWER
    # -------------------------
    st.subheader("Answer")

    st.write(response.content)