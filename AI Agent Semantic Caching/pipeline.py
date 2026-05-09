import os
import redis
import numpy as np
from openai import OpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", None),
    decode_responses=False
)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.75))
INDEX_NAME = "semantic_cache"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CacheState(TypedDict):
    query: str
    embedding: Optional[list]
    cached_response: Optional[str]
    llm_response: Optional[str]
    cache_hit: bool


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def embed_query(state: CacheState) -> CacheState:
    """Convert the raw query string into a vector embedding."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=state["query"]
    )
    state["embedding"] = response.data[0].embedding
    return state


def similarity_search(state: CacheState) -> CacheState:
    query_vec = np.array(state["embedding"], dtype=np.float32).tobytes()

    try:
        results = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            "*=>[KNN 1 @embedding $vec AS score]",
            "PARAMS", "2", "vec", query_vec,
            "RETURN", "2", "response", "score",
            "DIALECT", "2"
        )
    except Exception as e:
        print(f"[similarity_search] Redis search failed: {e}")
        state["cache_hit"] = False
        return state

    if results[0] > 0:
        # results[2] is a flat list: [field, value, field, value, ...]
        # parse it into a dict to avoid hardcoded index assumptions
        fields = results[2]
        result_dict = {}
        for i in range(1, len(fields), 2):
            key = fields[i - 1].decode() if isinstance(fields[i - 1], bytes) else fields[i - 1]
            val = fields[i]
            result_dict[key] = val

        if "score" in result_dict and "response" in result_dict:
            score = float(result_dict["score"])
            similarity = 1 - score
            print(f"[similarity_search] similarity={similarity:.4f} threshold={SIMILARITY_THRESHOLD}")
            if similarity >= SIMILARITY_THRESHOLD:
                response_val = result_dict["response"]
                state["cached_response"] = response_val.decode() if isinstance(response_val, bytes) else response_val
                state["cache_hit"] = True
                return state

    state["cache_hit"] = False
    return state

def call_llm(state: CacheState) -> CacheState:
    """Call the LLM. Only reached on a cache miss."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": state["query"]}]
    )
    state["llm_response"] = response.choices[0].message.content
    return state


def update_cache(state: CacheState) -> CacheState:
    """
    Store the new query-response pair in Redis.
    Reuses the embedding already computed in embed_query.
    """
    vec = np.array(state["embedding"], dtype=np.float32).tobytes()

    # Use a hash of the query as the key to avoid collisions on long queries.
    import hashlib
    key = "cache:" + hashlib.md5(state["query"].encode()).hexdigest()

    r.hset(key, mapping={
        "query": state["query"],
        "response": state["llm_response"],
        "embedding": vec
    })
    return state


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_after_search(state: CacheState) -> str:
    return "cache_hit" if state["cache_hit"] else "llm_call"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_pipeline() -> object:
    graph = StateGraph(CacheState)

    graph.add_node("embed_query", embed_query)
    graph.add_node("similarity_search", similarity_search)
    graph.add_node("call_llm", call_llm)
    graph.add_node("update_cache", update_cache)

    graph.set_entry_point("embed_query")
    graph.add_edge("embed_query", "similarity_search")
    graph.add_conditional_edges(
        "similarity_search",
        route_after_search,
        {
            "cache_hit": END,
            "llm_call": "call_llm"
        }
    )
    graph.add_edge("call_llm", "update_cache")
    graph.add_edge("update_cache", END)

    return graph.compile()


pipeline = build_pipeline()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def query(user_input: str) -> dict:
    """
    Run a query through the semantic cache pipeline.

    Returns:
        dict with keys: response, cache_hit, query
    """
    initial_state: CacheState = {
        "query": user_input,
        "embedding": None,
        "cached_response": None,
        "llm_response": None,
        "cache_hit": False
    }

    result = pipeline.invoke(initial_state)

    return {
        "query": user_input,
        "response": result["cached_response"] if result["cache_hit"] else result["llm_response"],
        "cache_hit": result["cache_hit"]
    }


if __name__ == "__main__":
    test_queries = [
        "what happens if my EMI bounces?",
        "will I be charged if I miss my loan payment?",   # should hit cache after first
        "how do I dispute a transaction?",
        "I see a charge I don't recognize, what should I do?"  # should hit cache after first
    ]

    for q in test_queries:
        result = query(q)
        hit = "CACHE HIT" if result["cache_hit"] else "LLM CALL"
        print(f"\n[{hit}] {result['query']}")
        print(f"Response: {result['response'][:120]}...")