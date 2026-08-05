"""Shared configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for both Traditional and LCEL pipelines."""

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2
    log_level: str = "INFO"

    def require_api_key(self) -> str:
        if not self.openai_api_key or self.openai_api_key.startswith("sk-your-key"):
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and set a valid key."
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
