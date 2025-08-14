from typing import Optional, Dict, Any
from .config import settings
from .retriever import search_books
from .tools import get_summary_by_title
from openai import OpenAI

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = (
    "Ești un asistent care recomandă cărți. "
    "Fii concis, prietenos și specific. "
    "Dacă recomandarea nu se potrivește perfect, explică de ce tot ar putea fi interesantă."
)

def recommend_with_rag(user_query: str) -> Dict[str, Any]:
    candidates = search_books(user_query, top_k=3)
    if not candidates:
        return {"message": "Nu am găsit nicio recomandare pentru această cerere."}
    best = candidates[0]
    user_prompt = (
        f"Utilizatorul caută o carte pe tema: `{user_query}`.\n"
        f"Candidatul cel mai bun: {best['title']}.\n"
        f"Rezumat scurt: {best['summary']}\n"
        f"Te rog oferă o recomandare conversațională (max 5-7 rânduri), "
        f"apoi sugerează 1-2 teme cheie."
    )
    chat = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )
    reply = chat.choices[0].message.content
    full_summary = get_summary_by_title(best["title"])
    return {
        "title": best["title"],
        "short_summary": best["summary"],
        "message": reply,
        "full_summary": full_summary,
    }
