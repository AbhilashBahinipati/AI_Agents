"""
setup_index.py

Run this once before using the pipeline.
Creates the Redis vector index used for semantic similarity search.
"""

import os
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType


from dotenv import load_dotenv

load_dotenv()

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", None),
    decode_responses=False
)

INDEX_NAME = "semantic_cache"

# text-embedding-3-small outputs 1536 dimensions.
# If you switch embedding models, update DIM to match.
EMBEDDING_DIM = 1536


def create_index():
    # Drop existing index if it exists (useful during development)
    print(f"Checking for existing index '{INDEX_NAME}'...")
    try:
        r.ft(INDEX_NAME).dropindex(delete_documents=False)
        print(f"Dropped existing index: {INDEX_NAME}")
    except Exception:
        pass  # Index did not exist, nothing to drop

    schema = [
        TextField("query"),
        TextField("response"),
        VectorField(
            "embedding",
            # FLAT = brute-force exact search.
            # Switch to HNSW for approximate search at >100k entries.
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": EMBEDDING_DIM,
                "DISTANCE_METRIC": "COSINE"
            }
        )
    ]

    print(f"Creating index '{INDEX_NAME}'...")

    r.ft(INDEX_NAME).create_index(
        schema,
        definition=IndexDefinition(
            prefix=["cache:"],
            index_type=IndexType.HASH
        )
    )

    print(f"Index '{INDEX_NAME}' created successfully.")
    print(f"  Embedding dimensions : {EMBEDDING_DIM}")
    print(f"  Distance metric      : COSINE")
    print(f"  Algorithm            : FLAT")


if __name__ == "__main__":
    create_index()