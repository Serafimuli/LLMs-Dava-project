# src/retriever.py
from __future__ import annotations

from typing import List, Dict, Any
import chromadb
from openai import OpenAI

from config import settings

# OpenAI client for embeddings
client = OpenAI(api_key=settings.openai_api_key)


def get_collection(name: str = "book_summaries"):
    """
    Connect to a persistent Chroma collection.
    We request cosine space; if the collection was originally created with a different
    space, delete/recreate the collection to switch metrics.
    """
    db = chromadb.PersistentClient(path=settings.chroma_path)
    return db.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # cosine distance => distance = 1 - cosine_similarity
    )


def embed_query(q: str) -> list[float]:
    """
    Create an embedding for the user query using the configured OpenAI model.
    """
    e = client.embeddings.create(model=settings.openai_embed_model, input=q)
    return e.data[0].embedding


def search_books(user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top_k most similar books to the user_query from Chroma.
    No similarity threshold is applied; we return what Chroma considers closest.
    Results are enriched with both distance and derived similarity for transparency.
    """
    col = get_collection()
    emb = embed_query(user_query)
    res = col.query(
        query_embeddings=[emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    items: List[Dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i in range(len(ids)):
        dist = float(dists[i])
        sim = 1.0 - dist  # cosine_similarity = 1 - cosine_distance
        items.append({
            "id": ids[i],
            "title": metas[i]["title"],
            "summary": docs[i],
            "distance": dist,
            "similarity": sim,
        })

    # Sort best-first by similarity (optional; Chroma usually returns sorted already)
    items.sort(key=lambda x: x["similarity"], reverse=True)
    return items
