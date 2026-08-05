"""Structural tests for LCEL composition (no live LLM calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.shared.schemas import AnalysisReport, FeatureExtraction, SentimentAnalysis


def _fake_features(_: dict) -> FeatureExtraction:
    return FeatureExtraction(
        product_name="CloudSync Pro",
        features_mentioned=["sync"],
        issues_mentioned=["crash"],
        key_quotes=["deal-breaker"],
    )


def _fake_sentiment(_: dict) -> SentimentAnalysis:
    return SentimentAnalysis(
        overall_sentiment="mixed",
        confidence=0.9,
        urgency="high",
        themes=["reliability"],
        rationale="Critical bug plus praise for sync.",
    )


def _fake_report(_: dict) -> AnalysisReport:
    features = _fake_features({})
    sentiment = _fake_sentiment({})
    return AnalysisReport(
        executive_summary="Fix restore crashes before renewal season.",
        recommended_actions=["Patch restore path", "Review pricing comms"],
        owner_team="engineering",
        priority_score=9,
        features=features,
        sentiment=sentiment,
    )


@patch("src.lcel.multi_step_pipeline.build_llm")
def test_lcel_pipeline_composes_and_invokes(mock_build_llm: MagicMock) -> None:
    """Ensure the LCEL graph wires steps without calling a real model."""
    from langchain_core.runnables import RunnableLambda

    from src.lcel.multi_step_pipeline import build_lcel_pipeline

    # build_lcel_pipeline does: prompt | llm.with_structured_output(Model)
    # We replace with_structured_output so the pipe becomes prompt | fake_runnable.
    fake_llm = MagicMock()

    def structured(model_cls):  # noqa: ANN001
        mapping = {
            FeatureExtraction: _fake_features,
            SentimentAnalysis: _fake_sentiment,
            AnalysisReport: _fake_report,
        }
        return RunnableLambda(mapping[model_cls])

    fake_llm.with_structured_output.side_effect = structured
    mock_build_llm.return_value = fake_llm

    pipeline = build_lcel_pipeline()
    result = pipeline.invoke({"feedback": "sync is great but restore crashes"})

    assert isinstance(result, AnalysisReport)
    assert result.features.product_name == "CloudSync Pro"
    assert result.priority_score == 9
    assert result.owner_team == "engineering"


@patch("src.traditional.multi_step_chain.build_llm")
def test_traditional_processor_steps(mock_build_llm: MagicMock) -> None:
    from langchain_core.runnables import RunnableLambda

    from src.traditional.multi_step_chain import TraditionalMultiStepProcessor

    fake_llm = MagicMock()

    def structured(model_cls):  # noqa: ANN001
        mapping = {
            FeatureExtraction: _fake_features,
            SentimentAnalysis: _fake_sentiment,
            AnalysisReport: _fake_report,
        }
        return RunnableLambda(mapping[model_cls])

    fake_llm.with_structured_output.side_effect = structured
    mock_build_llm.return_value = fake_llm

    processor = TraditionalMultiStepProcessor()
    report, steps = processor.process("sync is great but restore crashes")

    assert [s.name for s in steps] == ["extract", "sentiment", "report"]
    assert report.sentiment.overall_sentiment == "mixed"
    assert all(s.elapsed_ms >= 0 for s in steps)
