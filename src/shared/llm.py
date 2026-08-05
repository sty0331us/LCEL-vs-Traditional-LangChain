"""Shared helpers for both pipeline styles."""

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def build_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Create a ChatOpenAI client from shared settings."""
    cfg = settings or get_settings()
    cfg.require_api_key()
    return ChatOpenAI(
        model=cfg.openai_model,
        temperature=cfg.openai_temperature,
        api_key=cfg.openai_api_key,
    )


def configure_logging(level: str | None = None) -> None:
    cfg = get_settings()
    logging.basicConfig(
        level=getattr(logging, level or cfg.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def model_to_json(model: Any) -> str:
    """Serialize a Pydantic model (v1/v2 compatible) to JSON text."""
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=2)
    return model.json(indent=2)
