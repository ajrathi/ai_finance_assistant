# Finnie — AI Finance Assistant

> **Educational AI, not financial advice.** Finnie is a multi-agent conversational system that provides personalized financial education through intelligent, context-aware conversations.

---

## Chat Interface

Finnie uses a **single unified chat interface** — think ChatGPT, but for financial education.

### How it works

```
┌────────────────────────────────────────────────────────────────────────┐
│  SIDEBAR                 │  MAIN CHAT AREA                             │
│  ─────────────────────── │  ─────────────────────────────────────────  │
│  👤 Profile settings     │  [user] What is compound interest?          │
│  📊 Portfolio upload     │                                             │
│  🎯 Goal setup           │  [Finnie] Compound interest is…             │
│  📈 Market overview      │     Sources (1)  ▾                          │
│  ⚙️  System status        │  ┌─────────────────────────────────────┐   │
│                           │  │  [Portfolio Pie Chart]               │  │
│                           │  │  [Portfolio Bar Chart]               │  │
│                           │  └─────────────────────────────────────┘  │
│                           │                                             │
│                           │  Suggested follow-ups:                      │
│                           │  [What is the Rule of 72?]                 │
│                           │  [How does inflation affect savings?]       │
│                           │                                             │
│                           │  ┌────────────────────────────────────┐    │
│                           │  │ Ask Finnie anything…             ↵ │    │
│                           │  └────────────────────────────────────┘    │
└───────────────────────────┴─────────────────────────────────────────────┘
```

### Key behaviours

- **Everything goes through the chat.** Users ask natural language questions; the orchestrator classifies intent and routes to the appropriate agent — no hardcoded UI logic.
- **Charts appear inline.** When the orchestrator detects a portfolio, goal-projection, or market-data response, it renders the relevant chart directly below the assistant bubble.
- **Sidebar provides data context.** Upload a portfolio CSV, define a savings goal, or check live market metrics in the sidebar — then reference them naturally in chat ("analyze my portfolio", "discuss my goal").
- **Follow-up suggestions** appear after each response to guide the conversation.
- **Conversation history** is forwarded to the orchestrator (last 10 turns) so agents can answer follow-up questions coherently.

### How orchestration works in the new flow

```
User types query ──► build_graph_input()
                         │  adds: user_profile, portfolio_data,
                         │        conversation_history (last 10 turns)
                         ▼
               LangGraph pipeline (run_query)
                         │
                  intent_classifier_node
                  (keyword match → Gemini fallback)
                         │
                  route_by_intent()
                         │
          ┌──────────────┼─────────────────────┐
          ▼              ▼                      ▼
    finance_qa     portfolio_node         tax_node …
          │              │                      │
          └──────────────┴──────── synthesizer ─┘
                                       │
                              disclaimer_injector
                                       │
                                 final_response
                                 sources
                                 disclaimers
                                       │
                     ◄── app.py renders text + chart ──
```

The UI layer (`app.py`) never makes intent decisions. It passes the raw query to the orchestrator and renders whatever the orchestrator returns.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI (single chat page)                      │
│         Sidebar: profile/portfolio/goals/market widgets                  │
│         Main:    scrollable chat history + chat_input                    │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │ User Query
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ORCHESTRATION LAYER                         │
│                                                                          │
│  profile_loader → intent_classifier → router                            │
│                          │                                               │
│        ┌────────┬─────────┼──────────┬──────────┬──────────┐            │
│        ▼        ▼         ▼          ▼          ▼          ▼            │
│   Finance Q&A Portfolio Market  Goal Plan  News Synth  Tax Edu          │
│        └────────┴─────────┴──────────┴──────────┴──────────┘            │
│                          │                                               │
│              response_synthesizer → disclaimer_injector → END           │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         CORE SERVICES                                    │
│                                                                          │
│  Gemini 2.0 Flash    FAISS Index         TTL Cache         yFinance      │
│  (60 req/min)       (35+ articles)      (30-min market)   (fallback)    │
│  Rate-limited       Semantic search      15-min news       No key needed │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query → AgentState assembly → Intent Classification (keyword + Gemini)
    → LangGraph Router → Agent Node(s) → RAG Retrieval (FAISS)
    → Gemini 2.0 Flash → Response Synthesis → Disclaimer Injection → UI
```

---

## The Six Agents

| Agent | Domain | RAG Categories |
|-------|--------|----------------|
| **Finance Q&A** | General financial concepts, definitions | All |
| **Portfolio Analysis** | Allocation, diversification, risk assessment | investing, market_concepts |
| **Market Analysis** | Live prices, indexes, sector data | market_concepts |
| **Goal Planning** | Retirement, savings targets, projections | basics, investing, retirement |
| **News Synthesizer** | Financial news summarization | market_concepts |
| **Tax Education** | Tax concepts, tax-advantaged accounts | taxes, retirement |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- `pip` or `uv`
- API keys (see below)

### 1. Clone and Create Environment

```bash
git clone <repository-url>
cd ai_finance_assistant
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key   # optional, yfinance used as fallback
```

**Getting API Keys:**
- **Google Gemini**: https://aistudio.google.com/app/apikey (free tier: 60 req/min)
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key (free tier: 25 req/day)

### 4. Build the Knowledge Base Index

On first run, build the FAISS vector index from the 35 curated articles:

```bash
python -m src.data.seed_articles
```

This takes 1–3 minutes on first run. The index is saved to `src/data/faiss_index/`.

### 5. Launch the App

```bash
streamlit run run.py
# OR
python run.py
```

The app opens at http://localhost:8501

---

## Project Structure

```
ai_finance_assistant/
├── run.py                         # Entry point
├── config.yaml                    # All configuration
├── requirements.txt               # Dependencies
├── .env.example                   # API key template
│
├── src/
│   ├── agents/                    # 6 specialized agents
│   │   ├── base_agent.py          # Abstract base with AgentOutput schema
│   │   ├── finance_qa_agent.py    # General financial education
│   │   ├── portfolio_agent.py     # Portfolio analysis
│   │   ├── market_agent.py        # Live market data
│   │   ├── goal_planning_agent.py # Goal planning + projections
│   │   ├── news_agent.py          # News summarization
│   │   └── tax_agent.py           # Tax education
│   │
│   ├── core/                      # Infrastructure
│   │   ├── config.py              # Typed config loader
│   │   ├── llm_client.py          # Gemini client (rate-limited, cached)
│   │   ├── embeddings.py          # Embedding model (Google + local fallback)
│   │   └── exceptions.py          # Custom exception hierarchy
│   │
│   ├── data/
│   │   ├── articles/              # 35 curated knowledge base articles
│   │   │   ├── basics/            # (8 articles)
│   │   │   ├── investing/         # (8 articles)
│   │   │   ├── retirement/        # (5 articles)
│   │   │   ├── taxes/             # (5 articles)
│   │   │   ├── budgeting/         # (3 articles)
│   │   │   └── market_concepts/   # (5 articles + 1 extra)
│   │   ├── faiss_index/           # Generated index (run seed_articles.py)
│   │   └── seed_articles.py       # Index builder script
│   │
│   ├── rag/                       # RAG pipeline
│   │   ├── chunker.py             # 512-token chunks, 50-token overlap
│   │   ├── indexer.py             # FAISS IndexFlatIP builder
│   │   ├── retriever.py           # MMR re-ranking + category filtering
│   │   ├── pipeline.py            # retrieve_and_format() orchestrator
│   │   └── schemas.py             # Document, Chunk, RetrievalResult types
│   │
│   ├── utils/                     # Shared utilities
│   │   ├── rate_limiter.py        # Token bucket (60 req/min)
│   │   ├── cache.py               # TTL + LRU cache
│   │   ├── retry.py               # Exponential backoff decorator
│   │   ├── formatters.py          # Markdown/number formatters
│   │   └── validators.py          # Input validation
│   │
│   ├── web_app/                   # Streamlit UI
│   │   ├── app.py                 # Main entry: unified chat page + onboarding
│   │   ├── session.py             # Session state (messages, portfolio, goals)
│   │   └── components/
│   │       ├── context_sidebar.py # Sidebar: portfolio/goal/market/profile widgets
│   │       ├── charts.py          # Plotly chart wrappers
│   │       ├── disclaimer_banner.py
│   │       └── source_attribution.py
│   │
│   └── workflow/                  # LangGraph orchestration
│       ├── state.py               # AgentState, UserProfile types
│       ├── intent_labels.py       # Intent enum + keyword map
│       ├── router.py              # IntentClassifier
│       ├── nodes.py               # All LangGraph node functions
│       └── graph.py               # StateGraph assembly + run_query()
│
└── tests/
    ├── conftest.py                # Shared fixtures
    ├── unit/                      # Isolated unit tests
    └── integration/               # Multi-component tests
```

---

## API Documentation

### `run_query(query, user_profile, portfolio_data, conversation_history)` → `dict`

The main programmatic interface to Finnie.

```python
from src.workflow.graph import run_query
from src.workflow.state import UserProfile

profile = UserProfile(knowledge_level="beginner", risk_tolerance="conservative")

# Single-turn
result = run_query(query="What is an index fund?", user_profile=profile)

# Multi-turn — pass prior chat turns for context
result = run_query(
    query="How do I buy one?",
    user_profile=profile,
    conversation_history=[
        {"role": "user",      "content": "What is an index fund?"},
        {"role": "assistant", "content": "An index fund tracks a market index…"},
    ],
)

print(result["final_response"])   # Markdown-formatted educational response
print(result["sources"])           # List of SourceAttribution objects
print(result["disclaimers"])       # List of applicable disclaimer strings
```

### `AgentState.initial(query, user_profile, portfolio_data)` → `AgentState`

Create an initial graph state:

```python
from src.workflow.state import AgentState, UserProfile

state = AgentState.initial(
    query="Analyze my portfolio",
    user_profile=UserProfile(knowledge_level="advanced"),
    portfolio_data={"holdings": [
        {"ticker": "AAPL", "shares": 10, "price_per_share": 190.0, "value": 1900.0}
    ]},
)
```

### `FAISSRetriever.retrieve(query, top_k, categories)` → `List[RetrievalResult]`

Direct access to the RAG retriever:

```python
from src.rag.retriever import get_retriever

retriever = get_retriever()
results = retriever.retrieve(
    query="compound interest",
    top_k=5,
    categories=["basics"],  # Optional category filter
)
for r in results:
    print(r.title, r.score, r.chunk.text[:100])
```

### `MarketDataFetcher.get_quote(symbol)` → `dict | None`

Fetch a stock quote:

```python
from src.agents.market_agent import MarketDataFetcher

fetcher = MarketDataFetcher()
quote = fetcher.get_quote("AAPL")
# {"symbol": "AAPL", "price": 190.5, "change": 1.2, "change_pct": 0.63}
```

---

## Configuration Reference (`config.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `llm` | `model` | `gemini-2.0-flash` | Gemini model ID |
| `llm.rate_limit` | `requests_per_minute` | `60` | Token bucket capacity |
| `llm.rate_limit` | `retry_attempts` | `3` | Max retry attempts |
| `cache` | `market_data_ttl` | `1800` | Market data cache TTL (seconds) |
| `cache` | `news_ttl` | `900` | News cache TTL (seconds) |
| `cache` | `llm_response_ttl` | `86400` | LLM response cache TTL (seconds) |
| `rag` | `chunk_size` | `512` | Token chunk size |
| `rag` | `top_k_final` | `5` | Results after MMR re-ranking |
| `rag` | `similarity_threshold` | `0.3` | Minimum FAISS score threshold |

---

## Usage Examples

### Example 1: Basic Q&A

```
User: What is compound interest?
Finnie: Compound interest is interest earned on both your principal AND
        the interest you've already earned...
        [Source: What is Compound Interest? (basics)]
        [Educational purposes only...]
```

### Example 2: Portfolio Analysis

Upload a CSV with columns: `ticker, shares, price_per_share`

```csv
ticker,shares,price_per_share
AAPL,10,190.00
MSFT,5,420.00
BND,50,72.00
VTI,20,240.00
```

### Example 3: Goal Planning

```
User: I want to save $1 million for retirement in 30 years.
      I have $50,000 now and can save $1,000/month.
Finnie: [Shows 3 projection scenarios: conservative 5%, moderate 7%, optimistic 9%]
        [Educational illustration disclaimer]
```

### Example 4: Multi-Domain Query

```
User: How do my tech stock gains affect my taxes?
Finnie: [Portfolio agent + Tax agent both invoked]
        [Synthesized response covering allocation AND capital gains concepts]
```

---

## Running Tests

```bash
# All tests
pytest

# Specific suite
pytest tests/unit/
pytest tests/integration/

# With coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Skip slow integration tests
pytest tests/unit/ -v
```

---

## Troubleshooting

### "GOOGLE_API_KEY is not set"
Add `GOOGLE_API_KEY=your_key` to `.env` in the project root.

### "No FAISS index found"
Run `python -m src.data.seed_articles` to build the knowledge base index.

### "Market data temporarily unavailable"
- Check `ALPHA_VANTAGE_API_KEY` in `.env` (optional — yfinance used as fallback)
- yfinance may have rate limits under heavy use — the app will show cached or fallback data

### Rate limit errors from Gemini
The token bucket limiter (60 req/min) handles this automatically with exponential backoff. If you see persistent rate limit errors, check your Google API usage quota.

### Streamlit import errors
Ensure you've activated your virtual environment and run `pip install -r requirements.txt`.

---

## Disclaimer

**Finnie is an educational tool only.** All content is for informational and educational purposes and does not constitute financial, investment, tax, or legal advice. Always consult qualified professionals (financial advisors, CPAs, attorneys) before making financial decisions. Past performance does not indicate future results. All investment projections are illustrative scenarios only, not predictions or guarantees.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.0 Flash |
| Orchestration | LangGraph (StateGraph) |
| Vector Database | FAISS (IndexFlatIP, cosine similarity) |
| Embeddings | Google text-embedding-004 / HuggingFace MiniLM fallback |
| Market Data | Alpha Vantage + yFinance |
| UI | Streamlit + Plotly |
| State | Streamlit session state |
| Caching | In-memory TTL + LRU |
| Rate Limiting | Token bucket (60 req/min) |
| Testing | pytest + pytest-cov |

---
title: Ai Finance Assistant
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

