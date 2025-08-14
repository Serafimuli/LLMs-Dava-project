from typing import Tuple
from .config import settings
from openai import OpenAI

_BAD_WORDS = {
    # keep it tiny & illustrative — extend as needed
    "idiot", "stupid", "hate", "kill"
}

client = OpenAI(api_key=settings.openai_api_key)

def basic_badword_flag(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _BAD_WORDS)

def check_openai_moderation(text: str) -> Tuple[bool, str]:
    # Returns (blocked, category_info)
    try:
        resp = client.moderations.create(
            model=settings.OPENAI_MODERATION_MODEL,
            input=text
        )
        res = resp.results[0]
        return (res.flagged, str(res.categories))
    except Exception as e:
        # Fail open to avoid blocking the app if moderation is unavailable
        return (False, "moderation_unavailable")
