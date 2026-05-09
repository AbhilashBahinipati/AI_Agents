# Agentic AI Design Patterns with LangGraph

A practical, production-oriented guide to the 7 core agentic AI design patterns — implemented with LangGraph using real-world scenarios from **banking, wealth management, investment banking, and private equity**.

Built for engineers and AI practitioners who want to go beyond single-prompt demos and understand how to architect multi-agent systems that work in production.

---

## What This Repository Covers

Each pattern is a self-contained Python script with:
- A real financial services scenario (not toy examples)
- Fully commented LangGraph implementation
- Streaming output so you see results as they generate
- A consistent state-graph structure you can adapt to your own domain

---

## Patterns at a Glance

| # | Pattern | Domain | Scenario |
|---|---------|--------|----------|
| 01 | Sequential Pipeline | Retail Banking | Loan application processing |
| 02 | Parallel Fan-out | Wealth Management | Portfolio multi-dimensional risk analysis |
| 03 | Router | Corporate Banking | Client query triage and specialist dispatch |
| 04 | Reflection Loop | Investment Banking | Equity research note with critic review |
| 05 | Hierarchical Supervisor | Investment Banking | M&A deal coordination across workstreams |
| 06 | Human-in-the-Loop | Wealth Management | High-value trade approval by RM |
| 07 | Plan and Execute | Private Equity | Investment due diligence with dynamic task planning |

---

## Setup

**Prerequisites**
- Python 3.10 or above
- An OpenAI API key (gpt-4o-mini is used by default)

**Install dependencies**

```bash
git clone https://github.com/AbhilashBahinipati/AI_Agents
pip install -r requirements.txt
```

**Configure environment**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
LLM_PROVIDER = 'openai'

```

**Run any pattern**

```bash
python sequential_agent.py
python critic_agent.py
# and so on
```


## LLM Configuration

The `config.py` file controls which LLM is used. Default is OpenAI:

```python
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
```

To switch to Ollama locally:

```bash
export LLM_PROVIDER=ollama
ollama pull llama3.2
python sequential_agent.py
```

---

## Pattern Selection Framework

See [PATTERNS_GUIDE.md](./PATTERNS_GUIDE.MD) for a detailed breakdown of each pattern including when to use it, when to avoid it, cost and latency tradeoffs, and a decision framework for selecting the right pattern for your use case.


