# LLMs-Dava-project

## Build and Run Instructions

### 1. Clone the repository
```sh
git clone https://github.com/Serafimuli/LLMs-Dava-project.git
cd LLMs-Dava-project
```

### 2. Create and activate a virtual environment (recommended)
```sh
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```sh
pip install -r .\requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root with your OpenAI API key and any other required settings:
```
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_PATH=./data/chroma
OPENAI_CHAT_MODEL=gpt-4.1-nano
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy
OPENAI_STT_MODEL=whisper-1
OPENAI_MODERATION_MODEL=omni-moderation-latest
OPENAI_IMAGE_MODEL=gpt-image-1
```

### 5. Run the Streamlit app
```sh
streamlit run src/app_streamlit.py
```

The app will open in your browser. Follow the UI to get book recommendations, generate audio, and more.