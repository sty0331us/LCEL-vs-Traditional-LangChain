"""Traditional LangChain chaining for multi-step feedback processing.

Style characteristics
---------------------
* Explicit, imperative orchestration in Python
* Each step is a separate chain; the developer wires outputs → inputs by hand
* Easy to debug step-by-step, but verbose and harder to compose / stream / batch
* Uses modern `with_structured_output` under the hood (LLMChain/SequentialChain
  are legacy) while preserving the *traditional imperative chaining pattern*
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable

from src.shared.llm import build_llm, model_to_json
from src.shared.prompts import EXTRACT_PROMPT, REPORT_PROMPT, SENTIMENT_PROMPT
from src.shared.schemas import AnalysisReport, FeatureExtraction, SentimentAnalysis

logger = logging.getLogger(__name__)


@dataclass
class TraditionalStepResult:
    """Per-step telemetry for side-by-side comparison."""

    name: str
    elapsed_ms: float
    output: Any


class TraditionalMultiStepProcessor:
    """Imperative multi-step pipeline: extract → sentiment → report."""

    def __init__(self) -> None:
        llm = build_llm()
        self.extract_chain: Runnable = EXTRACT_PROMPT | llm.with_structured_output(
            FeatureExtraction
        )
        self.sentiment_chain: Runnable = SENTIMENT_PROMPT | llm.with_structured_output(
            SentimentAnalysis
        )
        self.report_chain: Runnable = REPORT_PROMPT | llm.with_structured_output(
            AnalysisReport
        )

    def process(self, feedback: str) -> tuple[AnalysisReport, list[TraditionalStepResult]]:
        """Run the three steps sequentially with manual data plumbing."""
        steps: list[TraditionalStepResult] = []

        # --- Step 1: Feature extraction ---------------------------------
        t0 = time.perf_counter()
        features: FeatureExtraction = self.extract_chain.invoke({"feedback": feedback})
        steps.append(
            TraditionalStepResult("extract", (time.perf_counter() - t0) * 1000, features)
        )
        logger.info("Traditional step=extract product=%s", features.product_name)

        # --- Step 2: Sentiment analysis (manual hand-off) ---------------
        features_json = model_to_json(features)
        t1 = time.perf_counter()
        sentiment: SentimentAnalysis = self.sentiment_chain.invoke(
            {"features_json": features_json}
        )
        steps.append(
            TraditionalStepResult(
                "sentiment", (time.perf_counter() - t1) * 1000, sentiment
            )
        )
        logger.info(
            "Traditional step=sentiment overall=%s urgency=%s",
            sentiment.overall_sentiment,
            sentiment.urgency,
        )

        # --- Step 3: Final report (manual merge of prior outputs) -------
        sentiment_json = model_to_json(sentiment)
        t2 = time.perf_counter()
        partial_report: AnalysisReport = self.report_chain.invoke(
            {
                "feedback": feedback,
                "features_json": features_json,
                "sentiment_json": sentiment_json,
            }
        )
        # Preserve upstream structured objects so the report is complete
        # even if the final LLM omits nested fields.
        report = partial_report.model_copy(
            update={"features": features, "sentiment": sentiment}
        )
        steps.append(
            TraditionalStepResult("report", (time.perf_counter() - t2) * 1000, report)
        )
        logger.info(
            "Traditional step=report priority=%s owner=%s",
            report.priority_score,
            report.owner_team,
        )

        return report, steps


def build_traditional_processor() -> TraditionalMultiStepProcessor:
    return TraditionalMultiStepProcessor()
