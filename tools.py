import json

with open("book_summaries.json", "r", encoding="utf-8") as f:
    book_summaries = json.load(f)

def get_summary_by_title(title: str) -> str:
    for book in book_summaries:
        if book["title"].lower() == title.lower():
            return book["summary"]
    return "Rezumatul nu a fost găsit."
