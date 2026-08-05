"""Modern LCEL (LangChain Expression Language) multi-step pipeline.

Style characteristics
---------------------
* Declarative composition with the `|` pipe operator
* Data flow expressed as a graph of Runnables (assign / passthrough / parallel)
* First-class streaming, batching, async, and retries without rewriting orchestration
* Same prompts & schemas as the Traditional path — only the *wiring* differs
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from src.shared.llm import build_llm, model_to_json
from src.shared.prompts import EXTRACT_PROMPT, REPORT_PROMPT, SENTIMENT_PROMPT
from src.shared.schemas import AnalysisReport, FeatureExtraction, SentimentAnalysis

logger = logging.getLogger(__name__)


@dataclass
class LCELStepTrace:
    """Lightweight timing envelope around the composed pipeline."""

    name: str
    elapsed_ms: float
    output: Any


def _features_to_json(features: FeatureExtraction) -> str:
    return model_to_json(features)


def _sentiment_to_json(sentiment: SentimentAnalysis) -> str:
    return model_to_json(sentiment)


def _merge_report(payload: dict[str, Any]) -> AnalysisReport:
    """Attach upstream structured objects onto the final report."""
    report: AnalysisReport = payload["report"]
    return report.model_copy(
        update={
            "features": payload["features"],
            "sentiment": payload["sentiment"],
        }
    )


def build_lcel_pipeline() -> Runnable:
    """Compose extract → sentiment → report as a single LCEL Runnable.

    Input:  ``{"feedback": str}``
    Output: ``AnalysisReport``
    """
    llm = build_llm()

    extract = EXTRACT_PROMPT | llm.with_structured_output(FeatureExtraction)
    sentiment = SENTIMENT_PROMPT | llm.with_structured_output(SentimentAnalysis)
    report = REPORT_PROMPT | llm.with_structured_output(AnalysisReport)

    # Declarative graph:
    #   feedback ──► features ──► sentiment ──► report ──► AnalysisReport
    #        │            │            │
    #        └────────────┴────────────┴──► state carried by Passthrough.assign
    #
    # Each .assign(...) runnable receives the full accumulated state, so prompt
    # variables (feedback, features_json, sentiment_json) resolve automatically.
    pipeline: Runnable = (
        RunnablePassthrough.assign(features=extract)
        .assign(features_json=lambda x: _features_to_json(x["features"]))
        .assign(sentiment=sentiment)
        .assign(sentiment_json=lambda x: _sentiment_to_json(x["sentiment"]))
        .assign(report=report)
        | RunnableLambda(_merge_report)
    )
    return pipeline


class LCELMultiStepProcessor:
    """Thin façade that mirrors the Traditional processor API."""

    def __init__(self) -> None:
        self.pipeline = build_lcel_pipeline()

    def process(self, feedback: str) -> tuple[AnalysisReport, list[LCELStepTrace]]:
        t0 = time.perf_counter()
        report: AnalysisReport = self.pipeline.invoke({"feedback": feedback})
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "LCEL pipeline done priority=%s owner=%s elapsed_ms=%.1f",
            report.priority_score,
            report.owner_team,
            elapsed,
        )
        # LCEL runs as one composed unit; expose a single end-to-end trace.
        traces = [LCELStepTrace("lcel_pipeline", elapsed, report)]
        return report, traces

    async def aprocess(self, feedback: str) -> AnalysisReport:
        """Native async invoke — a key LCEL advantage over hand-rolled loops."""
        return await self.pipeline.ainvoke({"feedback": feedback})

    def batch(self, feedbacks: list[str]) -> list[AnalysisReport]:
        """Native batch invoke across many inputs."""
        return self.pipeline.batch([{"feedback": f} for f in feedbacks])


def build_lcel_processor() -> LCELMultiStepProcessor:
    return LCELMultiStepProcessor()
