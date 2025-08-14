import json

def get_summary_by_title(title: str, json_path: str = "book_summaries.json") -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        books = json.load(f)
    for b in books:
        if b["title"].strip().lower() == title.strip().lower():
            return b["summary"]
    return "Rezumatul nu a fost găsit."