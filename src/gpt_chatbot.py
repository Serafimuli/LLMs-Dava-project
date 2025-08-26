# src/gpt_chatbot.py
from __future__ import annotations

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from openai import OpenAI

from config import settings
from retriever import search_books  # returns top-k candidates already filtered by similarity
from tools import get_summary_by_title

client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = (
    "You are an assistant that recommends books. "
    "Choose the most suitable book for the user's request (from the given candidates), "
    "briefly explain why it fits, then call the tool `get_summary_by_title` "
    "to include the full summary. "
    "Keep the recommendation concise (max 5–7 lines), followed by the full summary."
)


def _format_candidates(cands: List[Dict[str, Any]]) -> str:
    """Format candidates for the LLM context."""
    lines = []
    for c in cands:
        # keep snippet short to reduce tokens
        snippet = (c.get("summary") or "").replace("\n", " ")[:300]
        lines.append(f"- Title: {c['title']}\n  Summary: {snippet}...")
    return "\n".join(lines)


def gpt_rerank(user_query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fallback re-ranker using the chat model. Returns the chosen candidate dict (adds 'why' when possible).
    Use only if the tool-calling path doesn't happen for any reason.
    """
    if not candidates:
        return {}

    prompt = (
        "Choose the most suitable book for the request below.\n"
        "Reply STRICTLY in JSON with this format:\n"
        '{ "title": "<exact title from candidates>", "why": "<short reason>" }\n\n'
        f"Request: {user_query}\n\n"
        f"Candidates:\n{_format_candidates(candidates)}"
    )

    resp = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": "Evaluate the candidates and select the best match."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    try:
        data = json.loads(content)
        chosen_title = (data.get("title") or "").strip().lower()
        for c in candidates:
            if c["title"].strip().lower() == chosen_title:
                c["why"] = data.get("why", "")
                return c
    except Exception:
        pass

    # Fallback to the first candidate (already sorted by similarity in retriever)
    best = candidates[0]
    best["why"] = "Highest semantic match among candidates."
    return best


@dataclass
class RecommendationResult:
    message: str
    title: str
    candidates: Optional[List[Dict[str, Any]]] = field(default=None)
    chosen_reason: Optional[str] = field(default=None)


def recommend_with_rag(user_query: str, return_debug: bool = False) -> RecommendationResult:
    """
    Main entry point used by the UI.
    1) Retrieves multiple candidates (RAG)
    2) Lets GPT choose and call the tool `get_summary_by_title`
    3) Returns final, polished assistant message + chosen title

    Returns:
        {
          "message": "<final text with full summary>",
          "title": "<chosen title>",
          # optional when return_debug=True:
          "candidates": [...], "chosen_reason": "..."
        }
    """
    # 1) Retrieve top-k (already thresholded in retriever via cosine similarity)
    candidates = search_books(user_query, top_k=5)

    if not candidates:
        return RecommendationResult(
            message=(
                "I couldn't find a close enough match. "
                "Can you provide more details (e.g. genre, period, key themes)?"
            ),
            title="",
        )

    # 2) Provide candidates to GPT and let it CHOOSE + CALL THE TOOL
    tools = [{
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returns the full summary of the book.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The EXACT title of the book (must match one of the candidates)."
                    }
                },
                "required": ["title"]
            }
        }
    }]

    user_content = (
        f"User request: {user_query}\n\n"
        "Choose ONLY ONE title from the candidates below that best fits, "
        "briefly explain why, then call the tool `get_summary_by_title` with the chosen title. "
        "The final answer should be concise (max 5–7 lines) and then include the full summary returned by the tool.\n\n"
        "CANDIDATES:\n"
        f"{_format_candidates(candidates)}"
    )

    first = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.4
    )

    assistant_msg = first.choices[0].message

    # Build conversation state for the second turn (after tool execution)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        assistant_msg,  # contains potential tool_calls and/or partial natural text
    ]

    chosen_title_from_tool: Optional[str] = None
    chosen_reason: Optional[str] = None

    # 3) Execute tool calls locally and feed results back
    if getattr(assistant_msg, "tool_calls", None):
        for tc in assistant_msg.tool_calls:
            if tc.function.name == "get_summary_by_title":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                title_arg = (args.get("title") or "").strip()
                if not title_arg:
                    title_arg = candidates[0]["title"]
                chosen_title_from_tool = title_arg

                full_summary = get_summary_by_title(title_arg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "get_summary_by_title",
                    "content": full_summary
                })

    # 4) Get FINAL polished answer (assistant now sees the tool outputs)
    final = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.5
    )
    answer = final.choices[0].message.content or ""

    result = RecommendationResult(
        message=answer,
        title=chosen_title_from_tool or candidates[0]["title"],
    )

    if return_debug:
        result.candidates = candidates
        if chosen_reason:
            result.chosen_reason = chosen_reason
    return result
