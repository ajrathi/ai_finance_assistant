"""Shared state definitions for the Finnie LangGraph workflow."""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


@dataclass
class UserProfile:
    """Persists across the session; updated after each interaction."""
    knowledge_level: str = "beginner"           # beginner | intermediate | advanced
    risk_tolerance: str = "moderate"            # conservative | moderate | aggressive
    goals: List[Dict] = field(default_factory=list)
    portfolio_snapshot: Optional[Dict] = None
    conversation_summary: str = ""


def _merge_agent_outputs(a: Dict, b: Dict) -> Dict:
    """Reducer for agent_outputs: merge dicts by key."""
    return {**a, **b}


def _extend_list(a: List, b: List) -> List:
    """Reducer for lists: extend."""
    return a + b


def _dedup_extend(a: List, b: List) -> List:
    """Reducer for sources: extend and deduplicate by title."""
    seen = {s.title if hasattr(s, "title") else str(s) for s in a}
    result = list(a)
    for item in b:
        key = item.title if hasattr(item, "title") else str(item)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


class FinnieState(TypedDict, total=False):
    """TypedDict schema for LangGraph StateGraph channels.

    Using TypedDict ensures all keys are preserved across every node transition.
    Fields with Annotated reducers are merged rather than replaced.
    """
    messages: List[BaseMessage]
    conversation_history: List[Dict]
    current_query: str
    user_profile: Any                                      # UserProfile dataclass
    intent: str
    active_agents: List[str]
    agent_outputs: Annotated[Dict, _merge_agent_outputs]   # merge partial updates
    retrieved_docs: List
    sources: Annotated[List, _dedup_extend]                # deduplicated extend
    market_data: Optional[Dict]
    portfolio_data: Optional[Dict]
    iteration_count: int
    error_state: Optional[str]
    final_response: Optional[str]
    disclaimers: List[str]


class AgentState(dict):
    """LangGraph state — a plain dict with typed keys.

    We use a plain TypedDict-compatible dict so LangGraph's StateGraph
    can merge partial updates via its built-in reducer logic.
    """

    @classmethod
    def initial(
        cls,
        query: str = "",
        user_profile: Optional[UserProfile] = None,
        portfolio_data: Optional[Dict] = None,
        messages: Optional[List[BaseMessage]] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> "AgentState":
        state = cls()
        state["messages"] = messages or []
        # Plain-dict conversation history for agent context (role/content pairs)
        state["conversation_history"] = conversation_history or []
        state["current_query"] = query
        state["user_profile"] = user_profile or UserProfile()
        state["intent"] = ""
        state["active_agents"] = []
        state["agent_outputs"] = {}
        state["retrieved_docs"] = []
        state["sources"] = []
        state["market_data"] = None
        state["portfolio_data"] = portfolio_data
        state["iteration_count"] = 0
        state["error_state"] = None
        state["final_response"] = None
        state["disclaimers"] = []
        return state
