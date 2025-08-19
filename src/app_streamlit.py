import io
import base64
import streamlit as st
from openai import OpenAI
from PIL import Image
from config import settings
from vector_store import upsert_books
from gpt_chatbot import recommend_with_rag
from moderation import basic_badword_flag, check_openai_moderation

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
st.write("Ask for a recommendation, get a conversational answer plus a full summary. "
         "Optional: listen to audio, upload your voice, or generate an image.")

# --- Input section ---
with st.form("query_form", clear_on_submit=False):
    q = st.text_input("What kind of book are you looking for?", placeholder="e.g.: magic and friendship")
    submit = st.form_submit_button("Recommend")

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
        st.warning("Your message contains inappropriate terms or is empty. Please rephrase.")
    else:
        with st.spinner("Searching for recommendation..."):
            result = recommend_with_rag(q)
            if "title" not in result:
                st.error(result["message"])
            else:
                st.subheader(f"📖 Recommendation: {result['title']}")
                st.write(result["message"])
                with st.expander("📘 Full summary"):
                    st.write(result["full_summary"])
                st.session_state["last_title"] = result["title"]
                st.session_state["last_text"]  = result["message"] + "\n\n" + result["full_summary"]

st.divider()

# --- TTS (Text-to-Speech) ---
st.subheader("🔊 Read me the recommendation (TTS)")
if "last_text" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        voice = st.selectbox("Voice", ["alloy", "verse", "sage"], index=0)
    with col2:
        tts_btn = st.button("Generate audio")
    if tts_btn:
        try:
            speech = client.audio.speech.create(
                model=settings.openai_tts_model,
                voice=voice,
                input=st.session_state["last_text"]
            )
            audio_bytes = speech.read()  # type: ignore
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error(f"OpenAI TTS failed. Make sure the model is enabled for your key. Details: {e}")
else:
    st.info("Ask for a recommendation first to generate audio.")

st.divider()

# --- STT (Speech-to-Text) ---
st.subheader("🎙️ Talk to Smart Librarian (STT)")
audio_file = st.file_uploader("Upload an audio file (mp3/wav/m4a/ogg)", type=["mp3", "wav", "m4a", "ogg"])
if audio_file is not None and st.button("Transcribe and recommend"):
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
        st.write(f"🗣️ You said: _{transcribed}_")

        # Run recommendation
        if guardrails(transcribed):
            st.warning("The transcribed text was blocked by filters. Please rephrase.")
        else:
            with st.spinner("Searching for recommendation..."):
                result = recommend_with_rag(transcribed)
                if "title" not in result:
                    st.error(result["message"])
                else:
                    st.subheader(f"📖 Recommendation: {result['title']}")
                    st.write(result["message"])
                    with st.expander("📘 Full summary"):
                        st.write(result["full_summary"])
                    st.session_state["last_title"] = result["title"]
                    st.session_state["last_text"]  = result["message"] + "\n\n" + result["full_summary"]
    except Exception as e:
        st.error("Transcription failed. Check the Whisper model and file format.")

st.divider()

# --- Image generation ---
st.subheader("🖼️ Generate a representative image")
prompt_hint = "suggestive cover in minimalist style"
img_col1, img_col2 = st.columns([2,1])
with img_col1:
    custom_prompt = st.text_input("Prompt (optional)", value=prompt_hint)
with img_col2:
    gen_btn = st.button("Generate image")

if gen_btn:
    title = st.session_state.get("last_title")
    if not title:
        st.info("Ask for a recommendation first to get the title.")
    else:
        try:
            prompt = (f"Representative image for the book '{title}'. "
                      f"Style: {custom_prompt}. No text, no logo.")
            img = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
                n=1
            )
            b64 = img.data[0].b64_json
            raw = base64.b64decode(b64)
            st.image(Image.open(io.BytesIO(raw)), caption=f"Generated for: {title}")
        except Exception as e:
            st.error("Image generation failed. Check access to the image model.")
st.subheader("🖼️ Generate a representative image")
prompt_hint = "suggestive cover in minimalist style"
img_col1, img_col2 = st.columns([2,1])
with img_col1:
    custom_prompt = st.text_input("Prompt (optional)", value=prompt_hint)
with img_col2:
    gen_btn = st.button("Generate image")

if gen_btn:
    title = st.session_state.get("last_title")
    if not title:
        st.info("Ask for a recommendation first to get the title.")
    else:
        try:
            prompt = (f"Representative image for the book '{title}'. "
                      f"Style: {custom_prompt}. No text, no logo.")
            img = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
                n=1
            )
            b64 = img.data[0].b64_json
            raw = base64.b64decode(b64)
            st.image(Image.open(io.BytesIO(raw)), caption=f"Generated for: {title}")
        except Exception as e:
            st.error("Image generation failed. Check access to the image model.")
