"""
benchmark.py

Runs a set of banking support queries through the pipeline twice:
  Pass 1 -- cold cache (all LLM calls)
  Pass 2 -- warm cache (semantic hits expected)

Prints a comparison table: latency, LLM calls, cache hit rate, estimated cost.
"""

import time
import os
from pipeline import query, r, INDEX_NAME
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Sample banking query dataset
# Grouped by semantic intent to test cache hit behaviour.
# ---------------------------------------------------------------------------

QUERIES = [
    # Intent: EMI / loan payment failure
    "what happens if my EMI bounces?",
    "will I be charged if I miss my loan payment?",
    "what is the penalty for a failed auto-debit?",

    # Intent: transaction dispute
    "how do I dispute a transaction?",
    "I see a charge I don't recognize, what should I do?",
    "can I raise a chargeback for an unknown debit?",

    # Intent: minimum balance
    "what is the minimum balance required in my savings account?",
    "how much money do I need to keep in my account?",
    "will I be charged if my balance goes below the limit?",

    # Intent: account linking
    "how do I link my salary account to my trading account?",
    "can I connect two bank accounts together?",
    "how to link accounts for funds transfer?",

    # Intent: statement / billing cycle
    "when does my monthly statement get generated?",
    "what is my account statement date?",
    "how do I get my bank statement for last month?",
]

COST_PER_1K_INPUT_TOKENS = 0.005    # gpt-4o approximate
AVG_TOKENS_PER_QUERY = 250          # rough estimate


def flush_cache():
    """Delete all keys with the cache: prefix."""
    keys = r.keys("cache:*")
    if keys:
        r.delete(*keys)
    print(f"Cache flushed ({len(keys)} entries removed).\n")


def run_pass(label: str) -> dict:
    print(f"--- {label} ---")
    total_latency = 0
    llm_calls = 0
    cache_hits = 0

    for q in QUERIES:
        start = time.time()
        result = query(q)
        elapsed = time.time() - start

        total_latency += elapsed
        if result["cache_hit"]:
            cache_hits += 1
            status = "HIT"
        else:
            llm_calls += 1
            status = "MISS"

        print(f"  [{status}] ({elapsed:.2f}s) {q[:60]}")

    n = len(QUERIES)
    avg_latency = total_latency / n
    hit_rate = (cache_hits / n) * 100
    estimated_cost = llm_calls * AVG_TOKENS_PER_QUERY / 1000 * COST_PER_1K_INPUT_TOKENS

    return {
        "label": label,
        "avg_latency": avg_latency,
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
        "hit_rate": hit_rate,
        "estimated_cost": estimated_cost
    }


def print_comparison(cold: dict, warm: dict):
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Cold Cache':>15} {'Warm Cache':>15}")
    print("=" * 60)
    print(f"{'Avg Latency':<25} {cold['avg_latency']:>14.2f}s {warm['avg_latency']:>14.2f}s")
    print(f"{'LLM Calls':<25} {cold['llm_calls']:>15} {warm['llm_calls']:>15}")
    print(f"{'Cache Hits':<25} {cold['cache_hits']:>15} {warm['cache_hits']:>15}")
    print(f"{'Hit Rate':<25} {cold['hit_rate']:>14.1f}% {warm['hit_rate']:>14.1f}%")
    print(f"{'Estimated Cost':<25} ${cold['estimated_cost']:>14.4f} ${warm['estimated_cost']:>14.4f}")
    print("=" * 60)

    latency_improvement = ((cold["avg_latency"] - warm["avg_latency"]) / cold["avg_latency"]) * 100
    cost_savings = ((cold["estimated_cost"] - warm["estimated_cost"]) / max(cold["estimated_cost"], 0.0001)) * 100
    print(f"\nLatency improvement : {latency_improvement:.1f}%")
    print(f"Cost reduction      : {cost_savings:.1f}%")


if __name__ == "__main__":
    print("Semantic Cache Benchmark")
    print(f"Queries: {len(QUERIES)} | Threshold: {os.getenv('SIMILARITY_THRESHOLD', 0.75)}\n")

    flush_cache()

    cold = run_pass("Pass 1: Cold Cache")
    print()
    warm = run_pass("Pass 2: Warm Cache")

    print_comparison(cold, warm)