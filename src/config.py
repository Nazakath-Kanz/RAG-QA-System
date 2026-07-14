# src/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger

class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    GOOGLE_API_KEY: str
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Explicitly point to the .env file in the parent root directory
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
        env_file_encoding='utf-8',
        extra='ignore'  # Prevents crashes if extra variables exist in your environment
    )

try:
    settings = Settings()
    logger.info("Successfully validated and loaded runtime configurations.")
except Exception as e:
    logger.critical(f"Configuration initialization aborted. Check your .env file schema: {e}")
    raise e