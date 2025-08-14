import json, os, re
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from .config import settings

load_dotenv()
client = OpenAI(api_key=settings.openai_api_key)

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def make_chroma():
    os.makedirs(settings.chroma_path, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_path)

def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=settings.openai_embed_model, input=texts)
    return [d.embedding for d in resp.data]

def upsert_books(json_path: str, collection_name: str = "book_summaries") -> int:
    db = make_chroma()
    col = db.get_or_create_collection(name=collection_name)
    with open(json_path, "r", encoding="utf-8") as f:
        books = json.load(f)

    # Prepare payloads
    ids, docs, metadatas = [], [], []
    for b in books:
        ids.append(f"{_slug(b['title'])}")
        docs.append(b["summary"])
        metadatas.append({"title": b["title"]})

    # Check existing
    existing = set(col.get(ids=ids)["ids"])
    new_indices = [i for i, _id in enumerate(ids) if _id not in existing]
    if not new_indices:
        return 0

    new_docs = [docs[i] for i in new_indices]
    new_ids  = [ids[i] for i in new_indices]
    new_meta = [metadatas[i] for i in new_indices]
    new_emb  = embed_texts(new_docs)

    col.add(documents=new_docs, embeddings=new_emb, metadatas=new_meta, ids=new_ids)
    return len(new_indices)

if __name__ == "__main__":
    count = upsert_books("book_summaries.json")
    print(f"Upserted {count} new books.")
