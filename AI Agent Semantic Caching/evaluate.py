"""
evaluate.py

Evaluates semantic cache quality on a labeled test set.
Computes precision, recall, and prints a confusion matrix.

Run setup_index.py before running this script.
"""

import os
import hashlib
import numpy as np
from openai import OpenAI
from pipeline import query as run_query, r
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Labeled test dataset
# ---------------------------------------------------------------------------

SEED_QUERIES = [
    {"query": "what happens if my EMI bounces?",                       "intent": "emi_failure"},
    {"query": "how do I dispute a transaction?",                        "intent": "dispute"},
    {"query": "what is the minimum balance for my savings account?",    "intent": "min_balance"},
    {"query": "how do I link my salary account to my trading account?", "intent": "account_linking"},
    {"query": "when does my monthly statement get generated?",          "intent": "statement"},
]

TEST_CASES = [
    # True positives -- semantically similar, should hit cache
    {"query": "will I be penalised if my loan EMI fails?",             "should_hit": True},
    {"query": "I see an unknown charge, how do I raise a dispute?",     "should_hit": True},
    {"query": "how much balance must I maintain to avoid charges?",     "should_hit": True},
    {"query": "can I connect two of my bank accounts?",                 "should_hit": True},
    {"query": "what date does my bank statement get generated?",        "should_hit": True},

    # True negatives -- different intent, should NOT hit
    {"query": "how do I apply for a new credit card?",                  "should_hit": False},
    {"query": "what is the current home loan interest rate?",           "should_hit": False},
    {"query": "how do I reset my net banking password?",                "should_hit": False},
]


def get_embedding(text: str) -> list:
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def seed_cache():
    print("Seeding cache...")
    for item in SEED_QUERIES:
        vec = np.array(get_embedding(item["query"]), dtype=np.float32).tobytes()
        key = "cache:" + hashlib.md5(item["query"].encode()).hexdigest()
        r.hset(key, mapping={
            "query": item["query"],
            "response": f"[Seeded response for intent: {item['intent']}]",
            "embedding": vec
        })
    print(f"  {len(SEED_QUERIES)} seed entries added.\n")


def evaluate():
    tp, fp, tn, fn = [], [], [], []
    fp_cases, fn_cases = [], []

    for case in TEST_CASES:
        result = run_query(case["query"])
        hit = result["cache_hit"]
        should = case["should_hit"]

        if hit and should:
            tp.append(case)
        elif hit and not should:
            fp.append(case)
            fp_cases.append(case)
        elif not hit and not should:
            tn.append(case)
        else:
            fn.append(case)
            fn_cases.append(case)

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("Confusion Matrix")
    print("=" * 45)
    print(f"  True Positives  (correct hit)  : {len(tp)}")
    print(f"  False Positives (wrong hit)    : {len(fp)}")
    print(f"  True Negatives  (correct miss) : {len(tn)}")
    print(f"  False Negatives (missed hit)   : {len(fn)}")
    print("=" * 45)
    print(f"  Precision : {precision:.2%}")
    print(f"  Recall    : {recall:.2%}")
    print(f"  F1 Score  : {f1:.2%}")

    if fp_cases:
        print("\nFalse Positives -- consider raising SIMILARITY_THRESHOLD:")
        for case in fp_cases:
            print(f"  {case['query']}")

    if fn_cases:
        print("\nFalse Negatives -- consider lowering SIMILARITY_THRESHOLD:")
        for case in fn_cases:
            print(f"  {case['query']}")


if __name__ == "__main__":
    keys = r.keys("cache:*")
    if keys:
        r.delete(*keys)
        print(f"Cache flushed ({len(keys)} entries removed).")

    seed_cache()
    evaluate()