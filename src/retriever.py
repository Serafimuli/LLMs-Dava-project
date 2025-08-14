from typing import Optional, List, Dict, Any
import chromadb
from .config import settings
from openai import OpenAI

client = OpenAI(api_key=settings.openai_api_key)

def get_collection(name="book_summaries"):
    db = chromadb.PersistentClient(path=settings.chroma_path)
    return db.get_or_create_collection(name=name)
    
def embed_query(q: str) -> list[float]:
    e = client.embeddings.create(model=settings.openai_embed_model, input=q)
    return e.data[0].embedding

def search_books(user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    col = get_collection()
    emb = embed_query(user_query)
    res = col.query(query_embeddings=[emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    out = []
    for i in range(min(top_k, len(res["ids"][0]))):
        out.append({
            "id": res["ids"][0][i],
            "title": res["metadatas"][0][i]["title"],
            "summary": res["documents"][0][i],
            "distance": res["distances"][0][i]
        })
    return out
