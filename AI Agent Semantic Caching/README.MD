# Semantic Caching for Enterprise AI Agents

Companion code for the Medium article: **Semantic Caching for Enterprise AI Agents: Cut Costs, Kill Latency**

This repo implements a semantic cache pipeline using LangGraph and Redis Stack. Instead of calling the LLM on every user query, semantically similar queries are served from cache -- cutting latency by ~8x and LLM costs by ~70% on typical enterprise support workloads.

---

## Repo Structure

```
.
├── pipeline.py       # LangGraph pipeline: embed, search, route, cache
├── setup_index.py    # Create the Redis vector index (run once)
├── benchmark.py      # Cold vs warm cache performance comparison
├── evaluate.py       # Precision, recall, confusion matrix on labeled queries
├── requirements.txt
├── .env.example      # Copy to .env and fill in your values
└── .gitignore
```

---

## Prerequisites

- Python 3.10+
- Redis Stack (includes the vector search module)
- An OpenAI API key

**Install Redis Stack locally:**

```bash
# macOS
brew tap redis-stack/redis-stack
brew install redis-stack

redis-stack-server
```

```bash
# Ubuntu / Debian
curl -fsSL https://packages.redis.io/redis-stack/redis-stack-server-7.2.0-v10.focal.x86_64.tar.gz | tar xz
./redis-stack-server/bin/redis-stack-server
```

Or use the official Docker image if you prefer:

```bash
docker run -p 6379:6379 redis/redis-stack:latest
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/your-username/semantic-cache-agent.git
cd semantic-cache-agent
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**


Open `.env` and fill in your values:

```
OPENAI_API_KEY=your_openai_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
SIMILARITY_THRESHOLD=0.92
LLM_MODEL=gpt-4o
```

The `.env` file is listed in `.gitignore` and will never be committed. Do not hardcode credentials anywhere in the code.

**5. Create the Redis vector index**

```bash
python setup_index.py
```

This only needs to be run once. If you change the embedding model, run it again to rebuild the index with the correct dimensions.

---

## Usage

**Run a query through the pipeline:**

```bash
python pipeline.py
```

This runs a small set of banking support queries -- some will miss the cache (LLM call), subsequent semantically similar ones will hit it.

**Run the benchmark:**

```bash
python benchmark.py
```

Runs two passes over the query dataset. Pass 1 is cold cache (all LLM calls). Pass 2 is warm cache (semantic hits expected). Prints a comparison table.

**Run the quality evaluation:**

```bash
python evaluate.py
```

Seeds the cache with known queries, then evaluates precision and recall against a labeled test set. Prints a confusion matrix and highlights false positives and false negatives to help tune `SIMILARITY_THRESHOLD`.

---

## Tuning the Similarity Threshold

`SIMILARITY_THRESHOLD` controls the tradeoff between precision and recall:

| Threshold | Effect |
|-----------|--------|
| 0.85+ | Very strict. Few false positives, more cache misses. Good for legal/compliance content. |
| 0.75 | Recommended default for financial services. |
| 0.6 | Broader matching. Higher hit rate but more false positives. Better for general FAQ. |

Set it in `.env` and run `evaluate.py` to see the impact on your dataset before deploying.

---

## Switching Embedding Models

If you use a model other than `text-embedding-3-small`, update `EMBEDDING_DIM` in `setup_index.py` to match the model's output dimension, then re-run it to rebuild the index.

| Model | Dimensions |
|-------|-----------|
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |
| text-embedding-ada-002 | 1536 |

---

## Production Considerations

- Replace the hash-key prefix in `update_cache` with a proper UUID for high-throughput deployments.
- Switch the Redis index algorithm from `FLAT` to `HNSW` in `setup_index.py` once your cache exceeds ~100k entries.
- Add TTL to cache entries using `r.expire(key, seconds)` after the `hset` call in `update_cache`.
- Wire cache invalidation to your product change events so stale responses are flushed immediately.