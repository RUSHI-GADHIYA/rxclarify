"""Runtime configuration, loaded from environment / .env with an RX_ prefix."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/rxclarify/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DRUG_LIST_PATH = DATA_DIR / "drug_list.yml"

# BAAI/bge-small-en-v1.5 output width. Must match vector(N) in db/schema.sql.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RX_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://rx:rx@localhost:5433/rxclarify"

    # Generation provider. Both implement llm.base.LLMProvider, so the eval
    # harness in Phase 2 can swap between them without touching pipeline code.
    llm_provider: Literal["ollama", "bedrock"] = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    # Bedrock model IDs carry the `anthropic.` provider prefix.
    bedrock_model: str = "anthropic.claude-haiku-4-5"
    # AWS_REGION has no RX_ prefix — it is the standard AWS variable, read
    # directly rather than through this Settings class (see llm/bedrock.py).

    top_k: int = 6
    max_tokens: int = 1024

    # Optional; raises the openFDA daily quota from 1,000 to 120,000 requests.
    openfda_api_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
