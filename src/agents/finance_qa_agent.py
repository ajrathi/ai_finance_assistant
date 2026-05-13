"""Finance Q&A Agent — handles general financial education queries."""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.agents.base_agent import AgentOutput, BaseAgent, GENERAL_DISCLAIMER, SourceAttribution
from src.core.exceptions import AgentError

logger = logging.getLogger(__name__)

_SYSTEM_EXTRA = """
You specialize in general financial education: concepts, definitions, how-to explanations,
comparisons between financial products, and investment fundamentals.

Guidelines:
- Ground your answer in the retrieved context documents when available.
- Cite sources at the end using the Source labels provided.
- Offer 2-3 follow-up questions the user might find helpful.
- Never recommend specific securities, funds, or financial products by name as purchases.
- If a question falls outside general financial education, redirect politely.
"""

_PROMPT_TEMPLATE = """\
Retrieved educational context:
{context}

User question: {query}

Please provide a clear, educational answer.
At the end of your response, list 2-3 follow-up questions the user might want to ask, prefixed with "Follow-up:".
"""


class FinanceQAAgent(BaseAgent):
    """General financial education Q&A using RAG-augmented generation."""

    name = "Finance Q&A Agent"

    def __init__(self, llm_client=None, rag_pipeline=None):
        super().__init__(llm_client=llm_client)
        self._rag = rag_pipeline  # Injected; lazy-loaded if None

    def _get_rag(self):
        if self._rag is None:
            try:
                from src.rag.pipeline import get_rag_pipeline
                self._rag = get_rag_pipeline()
            except Exception:
                self._rag = None
        return self._rag

    def _retrieve_context(self, query: str) -> tuple[str, List[SourceAttribution]]:
        """Retrieve relevant chunks from FAISS. Returns (context_str, sources)."""
        rag = self._get_rag()
        if rag is None:
            return "", []
        try:
            result = rag.retrieve_and_format(query, categories=[])
            return result.context_text, result.sources
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return "", []

    def _parse_follow_ups(self, text: str) -> tuple[str, List[str]]:
        """Split follow-up questions out of LLM response."""
        lines = text.splitlines()
        follow_ups: List[str] = []
        body_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("follow-up:"):
                q = stripped[len("follow-up:"):].strip()
                if q:
                    follow_ups.append(q)
            else:
                body_lines.append(line)
        return "\n".join(body_lines).strip(), follow_ups

    def process(self, state: Any) -> AgentOutput:
        """Process the agent state and return an educational Q&A response."""
        try:
            query: str = state.get("current_query", "") if isinstance(state, dict) else getattr(state, "current_query", "")
            user_profile = state.get("user_profile", {}) if isinstance(state, dict) else getattr(state, "user_profile", {})
            knowledge_level = (
                user_profile.get("knowledge_level", "intermediate")
                if isinstance(user_profile, dict)
                else getattr(user_profile, "knowledge_level", "intermediate")
            )

            if not query.strip():
                return self._make_error_output("Empty query received.")

            context_text, sources = self._retrieve_context(query)
            system_prompt = self._build_system_prompt(knowledge_level, extra=_SYSTEM_EXTRA)
            user_prompt = _PROMPT_TEMPLATE.format(
                context=context_text or "No specific context retrieved — answer from general knowledge.",
                query=query,
            )

            raw_response = self._llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )

            content, follow_ups = self._parse_follow_ups(raw_response)

            return AgentOutput(
                content=content,
                agent_name=self.name,
                confidence=0.9 if context_text else 0.7,
                sources=sources,
                disclaimers=[GENERAL_DISCLAIMER],
                follow_up_questions=follow_ups[:3],
            )

        except Exception as exc:
            logger.error("FinanceQAAgent error: %s", exc, exc_info=True)
            return self._make_error_output(str(exc))
