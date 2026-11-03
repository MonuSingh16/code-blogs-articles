from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str = Field(min_length=1)
    llm_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 1024

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
