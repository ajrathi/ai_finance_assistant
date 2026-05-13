"""Abstract base class for all Finnie agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SourceAttribution:
    title: str
    category: str
    relevance_score: float
    excerpt: str = ""


@dataclass
class AgentOutput:
    content: str
    agent_name: str
    confidence: float = 1.0
    sources: List[SourceAttribution] = field(default_factory=list)
    disclaimers: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Knowledge-level prompt prefixes
_LEVEL_PREFIXES = {
    "beginner": (
        "You are talking to someone who is new to personal finance. "
        "Use simple, plain language. Avoid jargon. "
        "When you must use a financial term, briefly define it in parentheses."
    ),
    "intermediate": (
        "You are talking to someone with a basic understanding of personal finance. "
        "You can use common financial terms without defining every one."
    ),
    "advanced": (
        "You are talking to someone with solid financial knowledge. "
        "You can use industry terminology and discuss nuanced concepts."
    ),
}

GENERAL_DISCLAIMER = (
    "This information is for educational purposes only and does not constitute financial advice."
)


class BaseAgent(ABC):
    """All Finnie agents inherit from this class."""

    name: str = "BaseAgent"

    def __init__(self, llm_client=None):
        from src.core.llm_client import get_llm_client
        self._llm = llm_client or get_llm_client()

    def _knowledge_prefix(self, level: str) -> str:
        return _LEVEL_PREFIXES.get(level.lower(), _LEVEL_PREFIXES["intermediate"])

    def _build_system_prompt(self, knowledge_level: str, extra: str = "") -> str:
        base = (
            "You are Finnie, an AI financial education assistant. "
            "Your role is to educate users about personal finance — NOT to give specific financial advice. "
            "Always be clear that users should consult qualified professionals for personalised advice. "
            f"{self._knowledge_prefix(knowledge_level)}"
        )
        if extra:
            base += f"\n\n{extra}"
        return base

    @abstractmethod
    def process(self, state: Any) -> AgentOutput:
        """Process the current AgentState and return an AgentOutput."""
        ...

    def _make_error_output(self, error_msg: str) -> AgentOutput:
        return AgentOutput(
            content=(
                "I'm sorry, I encountered an issue generating a response. "
                "Please try again or rephrase your question."
            ),
            agent_name=self.name,
            confidence=0.0,
            disclaimers=[GENERAL_DISCLAIMER],
            metadata={"error": error_msg},
        )
