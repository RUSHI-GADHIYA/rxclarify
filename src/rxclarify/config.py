"""Runtime configuration, loaded from environment / .env with an RX_ prefix."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/rxclarify/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load .env into the real process environment.
#
# pydantic-settings reads .env only to populate the RX_-prefixed fields on
# Settings below; it does NOT export anything to os.environ. The third-party
# SDKs read the environment directly and know nothing about Settings —
# langchain-openai wants OPENAI_API_KEY, boto3 wants AWS_*. Without this call
# those keys sit in .env and are never seen.
#
# override=False so a variable already exported in the shell (or injected by
# CI, or an IAM role) wins over the file, which is the precedence you want
# everywhere except a developer laptop.
load_dotenv(REPO_ROOT / ".env", override=False)
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DRUG_LIST_PATH = DATA_DIR / "drug_list.yml"

# BAAI/bge-small-en-v1.5 output width. Must match vector(N) in db/schema.sql.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# List price in USD per million tokens, as (input, output). Used only to show an
# estimated cost per query in the UI.
#
# These are hardcoded and WILL go stale — update them whenever you change
# `openai_model` or `bedrock_model`, and treat the figure on screen as an
# estimate, not a bill. A model absent from this table simply shows no cost
# rather than a wrong one.
# Standard (short-context) rates, verified 2026-08-16. sol and terra also have a
# long-context tier at roughly double these rates; RxClarify prompts run ~2k
# tokens, so the standard tier always applies here.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (5.00, 30.00),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RX_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://rx:rx@localhost:5433/rxclarify"

    # Generation provider. Both resolve to a LangChain BaseChatModel, so the
    # LCEL chain and the Phase 2 eval harness swap between them by config alone.
    llm_provider: Literal["openai", "bedrock"] = "openai"

    openai_model: str = "gpt-5.6-luna"
    # GPT-5 family models are reasoning models, and langchain-openai strips
    # `temperature` for them — *unless* reasoning effort is "none". Grounded
    # citation from supplied excerpts is extraction, not reasoning, so "none"
    # is both the cheapest setting and the only one where temperature=0 is
    # actually honoured. Raising it silently discards temperature; see
    # llm/openai.py for the full constraint (including why there is no seed).
    openai_reasoning_effort: str = "none"
    # OPENAI_API_KEY carries no RX_ prefix — it is the standard variable name,
    # read straight from the environment by langchain-openai.

    # Bedrock model IDs carry the `anthropic.` provider prefix.
    bedrock_model: str = "anthropic.claude-haiku-4-5"
    # AWS_REGION likewise has no RX_ prefix — standard AWS variable.

    top_k: int = 6
    max_tokens: int = 1024

    @property
    def active_model(self) -> str:
        return self.bedrock_model if self.llm_provider == "bedrock" else self.openai_model

    @property
    def price_per_mtok(self) -> tuple[float, float] | None:
        """(input, output) USD per million tokens for the active model.

        None when the model is not in PRICING — the caller should then omit the
        cost readout rather than display a wrong number.
        """
        return PRICING.get(self.active_model)

    # Optional; raises the openFDA daily quota from 1,000 to 120,000 requests.
    openfda_api_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
