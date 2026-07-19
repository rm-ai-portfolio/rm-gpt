from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_api_key: str
    llm_model: str
    embedding_model: str
    ollama_base_url: str
    ollama_embedding_url: str
    data_dir: str = "./data"
    sessions_file: str = "./data/sessions.json"
    faiss_dir: str = "./data/faiss_indexes"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        Path(self.data_dir).mkdir(exist_ok=True)
        Path(self.faiss_dir).mkdir(exist_ok=True)

settings = Settings()
settings.ensure_directories()
