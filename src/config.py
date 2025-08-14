from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    chroma_path: str = Field("./data/chroma", env="CHROMA_PATH")
    openai_chat_model: str = Field("gpt-4o", env="OPENAI_CHAT_MODEL")
    openai_embed_model: str = Field("text-embedding-3-small", env="OPENAI_EMBED_MODEL")
    openai_tts_model: str = Field("tts-1", env="OPENAI_TTS_MODEL")
    openai_tts_voice: str = Field("alloy", env="OPENAI_TTS_VOICE")
    openai_stt_model: str = Field("whisper-1", env="OPENAI_STT_MODEL")
    openai_moderation_model: str = Field("omni-moderation-latest", env="OPENAI_MODERATION_MODEL")

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
