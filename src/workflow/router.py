"""Intent classification and workflow routing for Finnie."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from src.workflow.intent_labels import Intent, KEYWORD_MAP

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT = """\
Classify the user's financial question into ONE of these intents:
- general_finance: General financial concepts, definitions, explanations
- portfolio_review: Questions about the user's own portfolio, holdings, allocation
- market_data: Questions about current market prices, indexes, stock quotes
- goal_planning: Financial goals, retirement planning, saving targets, projections
- news_summary: Financial news, current events, market news
- tax_education: Tax concepts, capital gains, IRA/401k/HSA, deductions
- multi_domain: Question spans two or more distinct domains above
- greeting: Just saying hello or greeting
- out_of_scope: Not related to personal finance

User question: {query}

Respond with ONLY a JSON object:
{{"intent": "<one of the above>", "confidence": <0.0-1.0>}}
"""


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    secondary_intents: List[Intent]


def _keyword_classify(query: str) -> Optional[Intent]:
    """Fast keyword-based classification — no LLM call needed."""
    q_lower = query.lower()
    for keyword, intent in KEYWORD_MAP.items():
        if keyword in q_lower:
            return intent
    return None


def _llm_classify(query: str, llm_client) -> Optional[IntentResult]:
    """Use Gemini for classification when keywords are insufficient."""
    try:
        prompt = _CLASSIFICATION_PROMPT.format(query=query)
        raw = llm_client.generate_structured(
            user_prompt=prompt,
            temperature=0.0,
        )
        # Extract JSON from response
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        intent_str = data.get("intent", "general_finance")
        confidence = float(data.get("confidence", 0.8))
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.GENERAL_FINANCE
        return IntentResult(intent=intent, confidence=confidence, secondary_intents=[])
    except Exception as exc:
        logger.warning("LLM classification failed: %s", exc)
        return None


class IntentClassifier:
    """Classifies user queries into intents using keyword matching + Gemini fallback."""

    def __init__(self, llm_client=None):
        self._llm = llm_client

    def _get_llm(self):
        if self._llm is None:
            from src.core.llm_client import get_llm_client
            self._llm = get_llm_client()
        return self._llm

    def classify(self, query: str) -> IntentResult:
        """Classify query. Returns IntentResult with primary intent and confidence."""
        if not query.strip():
            return IntentResult(
                intent=Intent.GREETING, confidence=1.0, secondary_intents=[]
            )

        # Fast keyword path
        keyword_intent = _keyword_classify(query)
        if keyword_intent is not None:
            logger.debug("Keyword classified as: %s", keyword_intent)
            return IntentResult(
                intent=keyword_intent, confidence=0.9, secondary_intents=[]
            )

        # LLM classification
        result = _llm_classify(query, self._get_llm())
        if result:
            logger.debug("LLM classified as: %s (%.2f)", result.intent, result.confidence)
            return result

        # Default fallback
        return IntentResult(
            intent=Intent.GENERAL_FINANCE, confidence=0.5, secondary_intents=[]
        )
