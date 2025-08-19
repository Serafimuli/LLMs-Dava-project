from typing import Optional, Dict, Any
from config import settings
from retriever import search_books
from tools import get_summary_by_title
from openai import OpenAI

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = (
    "You are an assistant that recommends books. "
    "Be concise, friendly, and specific. "
    "If the recommendation is not a perfect match, explain why it might still be interesting."
)

def recommend_with_rag(user_query: str) -> Dict[str, Any]:
    candidates = search_books(user_query, top_k=3)
    if not candidates:
        return {"message": "I couldn't find any recommendations for this request."}
    best = candidates[0]
    user_prompt = (
        f"The user is looking for a book on the topic: `{user_query}`.\n"
        f"Best candidate: {best['title']}.\n"
        f"Short summary: {best['summary']}\n"
        f"Please provide a conversational recommendation (max 5-7 lines), "
        f"then suggest 1-2 key themes."
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
