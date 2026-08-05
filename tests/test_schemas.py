"""Unit tests that do not require a live OpenAI API key."""

from __future__ import annotations

from src.shared.schemas import AnalysisReport, FeatureExtraction, SentimentAnalysis


def test_feature_extraction_schema() -> None:
    features = FeatureExtraction(
        product_name="CloudSync Pro",
        features_mentioned=["real-time sync"],
        issues_mentioned=["version restore crash"],
        key_quotes=["deal-breaker for my design workflow"],
    )
    assert features.product_name == "CloudSync Pro"
    assert len(features.issues_mentioned) == 1


def test_sentiment_schema_bounds() -> None:
    sentiment = SentimentAnalysis(
        overall_sentiment="mixed",
        confidence=0.82,
        urgency="high",
        themes=["reliability", "pricing"],
        rationale="Strong positives offset by critical restore bug and price shock.",
    )
    assert 0.0 <= sentiment.confidence <= 1.0
    assert sentiment.urgency == "high"


def test_analysis_report_composition() -> None:
    features = FeatureExtraction(product_name="NoteFlow")
    sentiment = SentimentAnalysis(
        overall_sentiment="negative",
        confidence=0.7,
        urgency="medium",
        themes=["performance"],
        rationale="Search latency dominates the experience.",
    )
    report = AnalysisReport(
        executive_summary="Search performance is blocking adoption.",
        recommended_actions=["Profile search index", "Ship offline conflict UX"],
        owner_team="engineering",
        priority_score=8,
        features=features,
        sentiment=sentiment,
    )
    payload = report.model_dump()
    assert payload["owner_team"] == "engineering"
    assert payload["features"]["product_name"] == "NoteFlow"


def test_prompts_export() -> None:
    from src.shared.prompts import EXTRACT_PROMPT, REPORT_PROMPT, SENTIMENT_PROMPT

    assert EXTRACT_PROMPT is not None
    assert SENTIMENT_PROMPT is not None
    assert REPORT_PROMPT is not None
