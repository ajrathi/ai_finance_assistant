# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Finnie** is an educational AI financial assistant (not financial advice) built with a multi-agent architecture. It uses Google Gemini 2.0 Flash + LangGraph orchestration + FAISS-based RAG to answer financial questions through a Streamlit UI.

## Commands

### Setup (first time)
```bash
# Install dependencies (prefer uv for speed)
uv pip install -r requirements.txt
# or: pip install -r requirements.txt

# Copy env template and add API keys
cp .env.example .env
# Required: GOOGLE_API_KEY
# Optional: ALPHA_VANTAGE_API_KEY (falls back to yFinance)

# Build FAISS knowledge base index (one-time, ~1-3 minutes)
python -m src.data.seed_articles
```

### Run the app
```bash
streamlit run run.py
# Opens at http://localhost:8501
```

### Tests
```bash
pytest                          # All tests (70% coverage required)
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only
pytest --cov=src --cov-report=html  # With HTML coverage report

# Single test file
pytest tests/unit/test_finance_qa_agent.py -v
```

## Architecture

### Request Flow
```
User query → LangGraph pipeline (src/workflow/graph.py run_query())
  → profile_loader_node
  → intent_classifier_node (keyword match + Gemini for ambiguous cases)
  → route_by_intent → one of 6 agent nodes
  → Agent.process(): RAG retrieval → prompt build → Gemini call → parse output
  → response_synthesizer_node
  → disclaimer_injector_node
  → profile_updater_node
  → final_response returned to Streamlit tab
```

### Key Layers

| Layer | Path | Purpose |
|-------|------|---------|
| Workflow/Orchestration | `src/workflow/` | LangGraph state machine (`graph.py`), intent routing (`router.py`), node functions (`nodes.py`) |
| Agents | `src/agents/` | 6 domain agents extend `BaseAgent`; each calls RAG then Gemini |
| RAG Pipeline | `src/rag/` | Chunk → embed → FAISS index → MMR re-rank (top 10 → 5) |
| Core Infrastructure | `src/core/` | `config.py` (settings singleton), `llm_client.py` (rate-limited Gemini wrapper), `embeddings.py` |
| UI | `src/web_app/` | Streamlit app with 5 tabs (Chat, Portfolio, Market, Goals, Education) |
| Knowledge Base | `src/data/articles/` | 35 curated markdown articles across 6 categories |
| Utilities | `src/utils/` | Token-bucket rate limiter, TTL cache, `@with_retry` decorator |

### The 6 Agents
- **FinanceQA** — general concepts, definitions (accesses all RAG categories)
- **Portfolio** — allocation analysis (investing, market_concepts)
- **Market** — live prices via Alpha Vantage → yFinance fallback (market_concepts)
- **GoalPlanning** — retirement/savings projections with 3 scenarios 5%/7%/9% (basics, investing, retirement)
- **News** — financial news summarization (market_concepts)
- **Tax** — tax concepts and tax-advantaged accounts (taxes, retirement)

### State & Configuration
- `AgentState` (LangGraph state dict) and `UserProfile` dataclass defined in `src/workflow/state.py`
- All settings in `config.yaml` — LLM model, rate limits, cache TTLs, RAG params, per-agent RAG category filters
- `src/core/config.py` exposes a `settings` singleton that merges `config.yaml` + `.env`
- FAISS index is built once by `seed_articles.py` and loaded at startup

### Caching Strategy
- Market data: 30 min TTL
- News: 15 min TTL
- LLM responses: 24 hr TTL
- FAISS retrieval results: 1 hr TTL
- Rate limiter: token bucket at 60 req/min (Gemini free tier)

### Knowledge-Level Adaptation
All agents adapt prompts based on `UserProfile.knowledge_level`: beginner (jargon explained), intermediate (common terms), advanced (industry terminology).

## External Dependencies
- **Google Gemini 2.0 Flash** (`GOOGLE_API_KEY`) — LLM + embeddings (text-embedding-004); required
- **Alpha Vantage** (`ALPHA_VANTAGE_API_KEY`) — live stock data (25 req/day free tier); optional, yFinance is fallback
- **yFinance** — no key needed, used as Alpha Vantage fallback
