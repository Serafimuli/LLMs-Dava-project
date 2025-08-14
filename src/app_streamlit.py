import io
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from .config import settings
from .vector_store import upsert_books
from .gpt_chatbot import recommend_with_rag
from .moderation import basic_badword_flag, check_openai_moderation

load_dotenv()
client = OpenAI(api_key=settings.openai_api_key)

st.set_page_config(page_title="Smart Librarian", page_icon="📚", layout="centered")

# --- Sidebar: one-time init ---
with st.sidebar:
    st.header("⚙️ Setup")
    if st.button("(Re)index book_summaries.json into Chroma"):
        n = upsert_books("book_summaries.json")
        st.success(f"Upserted {n} new books. (Data persisted at {settings.chroma_path})")

    st.caption(f"Chroma path: `{settings.chroma_path}`")
    st.caption(f"Models: chat={settings.openai_chat_model}, emb={settings.openai_embed_model}")

st.title("📚 Smart Librarian — RAG + Tools")
st.write("Cere o recomandare, primești un răspuns conversațional + rezumat complet. "
         "Opțional: ascultă audio, încarcă voce, generează imagine.")

# --- Input section ---
with st.form("query_form", clear_on_submit=False):
    q = st.text_input("Ce fel de carte cauți?", placeholder="ex: magie și prietenie")
    submit = st.form_submit_button("Recomandă")

def guardrails(text: str) -> bool:
    # True = block
    if not text or text.strip() == "":
        return True
    if basic_badword_flag(text):
        return True
    flagged, _cats = check_openai_moderation(text)
    return flagged

if submit:
    if guardrails(q):
        st.warning("Mesajul tău conține termeni nepotriviți sau este gol. Te rog reformulează.")
    else:
        with st.spinner("Caut recomandarea..."):
            result = recommend_with_rag(q)
        if "title" not in result:
            st.error(result["message"])
        else:
            st.subheader(f"📖 Recomandare: {result['title']}")
            st.write(result["message"])
            with st.expander("📘 Rezumat complet"):
                st.write(result["full_summary"])
            st.session_state["last_title"] = result["title"]
            st.session_state["last_text"]  = result["message"] + "\n\n" + result["full_summary"]

st.divider()

# --- TTS (Text-to-Speech) ---
st.subheader("🔊 Citește-mi recomandarea (TTS)")
if "last_text" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        voice = st.selectbox("Voce", ["alloy", "verse", "sage"], index=0)
    with col2:
        tts_btn = st.button("Generează audio")
    if tts_btn:
        try:
            speech = client.audio.speech.create(
                model=settings.openai_tts_model,
                voice=voice,
                input=st.session_state["last_text"],
                format="mp3"
            )
            audio_bytes = speech.read()  # type: ignore
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error("OpenAI TTS a eșuat. Asigură-te că modelul este activ pentru cheia ta.")
else:
    st.info("Cere mai întâi o recomandare pentru a genera audio.")

st.divider()

# --- STT (Speech-to-Text) ---
st.subheader("🎙️ Vorbește cu Smart Librarian (STT)")
audio_file = st.file_uploader("Încarcă un fișier audio (mp3/wav/m4a/ogg)", type=["mp3", "wav", "m4a", "ogg"])
if audio_file is not None and st.button("Transcrie și recomandă"):
    try:
        # Send to Whisper for transcription
        audio_bytes = audio_file.read()
        with io.BytesIO(audio_bytes) as f:
            f.name = audio_file.name  # provide a filename hint
            transcript = client.audio.transcriptions.create(
                model=settings.openai_stt_model,
                file=f
            )
        transcribed = transcript.text
        st.write(f"🗣️ Ai spus: _{transcribed}_")

        # Run recommendation
        if guardrails(transcribed):
            st.warning("Textul transcris a fost blocat de filtre. Reformulează.")
        else:
            with st.spinner("Caut recomandarea..."):
                result = recommend_with_rag(transcribed)
            if "title" not in result:
                st.error(result["message"])
            else:
                st.subheader(f"📖 Recomandare: {result['title']}")
                st.write(result["message"])
                with st.expander("📘 Rezumat complet"):
                    st.write(result["full_summary"])
                st.session_state["last_title"] = result["title"]
                st.session_state["last_text"]  = result["message"] + "\n\n" + result["full_summary"]
    except Exception as e:
        st.error("Transcrierea a eșuat. Verifică modelul Whisper și formatul fișierului.")

st.divider()

# --- Image generation ---
st.subheader("🖼️ Generează o imagine reprezentativă")
prompt_hint = "copertă sugestivă în stil minimalist"
img_col1, img_col2 = st.columns([2,1])
with img_col1:
    custom_prompt = st.text_input("Prompt (opțional)", value=prompt_hint)
with img_col2:
    gen_btn = st.button("Generează imagine")

if gen_btn:
    title = st.session_state.get("last_title")
    if not title:
        st.info("Cere mai întâi o recomandare pentru a ști titlul.")
    else:
        try:
            prompt = (f"Imagine reprezentativă pentru cartea '{title}'. "
                      f"Stil: {custom_prompt}. Fără text, fără logo.")
            img = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
                n=1
            )
            b64 = img.data[0].b64_json
            raw = base64.b64decode(b64)
            st.image(Image.open(io.BytesIO(raw)), caption=f"Generat pentru: {title}")
        except Exception as e:
            st.error("Generarea imaginii a eșuat. Verifică accesul la modelul de imagini.")
