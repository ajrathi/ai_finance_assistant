"""Tax Education Agent — explains tax concepts and account types."""
from __future__ import annotations

import logging
from typing import Any, List

from src.agents.base_agent import AgentOutput, BaseAgent, SourceAttribution

logger = logging.getLogger(__name__)

TAX_DISCLAIMER = (
    "Tax information provided is for educational purposes only. "
    "Consult a qualified tax professional for advice specific to your situation."
)

_SYSTEM_EXTRA = """
You specialize in tax education for personal finance. You explain:
- Income tax concepts (brackets, deductions, credits)
- Capital gains taxes (short-term vs long-term)
- Tax-advantaged accounts (401k, IRA, HSA, 529)
- Self-employment taxes
- Tax-loss harvesting concepts
- Estate and inheritance tax basics

Guidelines:
- Explain tax concepts clearly and accurately
- Always recommend consulting a qualified tax professional for personal situations
- Never provide specific tax filing advice
- Use current year's tax rules when citing specific numbers
- Make clear when rules are complex or situational
"""

_PROMPT_TEMPLATE = """\
Educational tax context:
{context}

User question about taxes: {query}

Please provide an educational response covering the relevant tax concepts.
Be accurate, clear, and always remind the user to consult a tax professional for their specific situation.
Use current U.S. tax rules.

End with 2-3 educational follow-ups prefixed "Follow-up:"
"""


class TaxEducationAgent(BaseAgent):
    """Tax concepts and account education agent."""

    name = "Tax Education Agent"

    def __init__(self, llm_client=None, rag_pipeline=None):
        super().__init__(llm_client=llm_client)
        self._rag = rag_pipeline

    def _get_rag(self):
        if self._rag is None:
            try:
                from src.rag.pipeline import get_rag_pipeline
                self._rag = get_rag_pipeline()
            except Exception:
                self._rag = None
        return self._rag

    def _retrieve_context(self, query: str) -> tuple[str, List[SourceAttribution]]:
        rag = self._get_rag()
        if rag is None:
            return "", []
        try:
            from src.core.config import settings
            categories = settings.agents.tax.rag_categories
            result = rag.retrieve_and_format(query, categories=categories)
            return result.context_text, result.sources
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return "", []

    def _parse_follow_ups(self, text: str) -> tuple[str, List[str]]:
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
        try:
            if isinstance(state, dict):
                query = state.get("current_query", "")
                user_profile = state.get("user_profile", {})
            else:
                query = getattr(state, "current_query", "")
                user_profile = getattr(state, "user_profile", {})

            knowledge_level = (
                user_profile.get("knowledge_level", "intermediate")
                if isinstance(user_profile, dict)
                else getattr(user_profile, "knowledge_level", "intermediate")
            )

            context_text, sources = self._retrieve_context(
                query or "tax brackets capital gains IRA 401k"
            )

            system_prompt = self._build_system_prompt(knowledge_level, extra=_SYSTEM_EXTRA)
            user_prompt = _PROMPT_TEMPLATE.format(
                context=context_text or "No specific context retrieved — answer from general knowledge.",
                query=query or "Explain key tax concepts for investors.",
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
                confidence=0.9 if context_text else 0.75,
                sources=sources,
                disclaimers=[TAX_DISCLAIMER],
                follow_up_questions=follow_ups[:3],
            )

        except Exception as exc:
            logger.error("TaxEducationAgent error: %s", exc, exc_info=True)
            return self._make_error_output(str(exc))
