import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Reconnect to the existing vector store
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="book_summaries")

def embed_query(query: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def search_books(user_query: str, top_k: int = 1):
    embedding = embed_query(user_query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    if results and results['documents']:
        return {
            "title": results['metadatas'][0][0]['title'],
            "summary": results['documents'][0][0]
        }
    return None