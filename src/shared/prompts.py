"""Prompt templates shared by Traditional and LCEL implementations.

Keeping prompts identical isolates the comparison to *composition style*,
not wording differences.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a product-feedback analyst. Extract structured signal from "
            "customer text. Be precise and prefer short, concrete phrases. "
            "Return only the requested fields.",
        ),
        (
            "human",
            "Customer feedback:\n\n{feedback}\n\n"
            "Extract product name, features mentioned, issues mentioned, and key quotes.",
        ),
    ]
)

SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a sentiment and priority analyst for a SaaS product team. "
            "Use the extracted features to judge overall sentiment, urgency, and themes.",
        ),
        (
            "human",
            "Extracted features (JSON):\n{features_json}\n\n"
            "Produce overall_sentiment, confidence (0-1), urgency, themes, and rationale.",
        ),
    ]
)

REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a staff product manager. Turn extraction + sentiment into an "
            "actionable internal report. Be concise and operational.",
        ),
        (
            "human",
            "Original feedback:\n{feedback}\n\n"
            "Features (JSON):\n{features_json}\n\n"
            "Sentiment (JSON):\n{sentiment_json}\n\n"
            "Write an executive_summary, recommended_actions, owner_team, and priority_score (1-10).",
        ),
    ]
)
