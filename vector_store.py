import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="book_summaries")

def load_books(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def store_books(json_file):
    books = load_books(json_file)
    for i, book in enumerate(books):
        embedding = embed_text(book["summary"])
        collection.add(
            documents=[book["summary"]],
            metadatas=[{"title": book["title"]}],
            ids=[f"book_{i}"]
        )
    print(f"Stored {len(books)} books into ChromaDB.")

if __name__ == "__main__":
    store_books("book_summaries.json")