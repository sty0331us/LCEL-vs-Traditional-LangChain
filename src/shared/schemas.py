"""Pydantic models shared by Traditional and LCEL pipelines."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeatureExtraction(BaseModel):
    """Step 1 — structured features pulled from raw customer feedback."""

    product_name: str = Field(description="Inferred or stated product name")
    features_mentioned: list[str] = Field(
        default_factory=list,
        description="Concrete product attributes mentioned by the customer",
    )
    issues_mentioned: list[str] = Field(
        default_factory=list,
        description="Problems, bugs, or friction points called out",
    )
    key_quotes: list[str] = Field(
        default_factory=list,
        description="Short verbatim quotes that carry strong signal",
    )


class SentimentAnalysis(BaseModel):
    """Step 2 — sentiment and priority derived from extracted features."""

    overall_sentiment: Literal["positive", "neutral", "negative", "mixed"]
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence 0–1")
    urgency: Literal["low", "medium", "high"]
    themes: list[str] = Field(default_factory=list, description="Recurring themes")
    rationale: str = Field(description="Brief reasoning for the sentiment call")


class AnalysisReport(BaseModel):
    """Step 3 — final multi-step output for downstream systems."""

    executive_summary: str
    recommended_actions: list[str] = Field(default_factory=list)
    owner_team: Literal["product", "engineering", "support", "marketing"]
    priority_score: int = Field(ge=1, le=10, description="1 = ignore, 10 = act now")
    features: FeatureExtraction
    sentiment: SentimentAnalysis
