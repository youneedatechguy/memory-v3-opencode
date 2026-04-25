"""
memory-v3 -- Embedding Layer
Delegates to the configured provider via get_embedder() factory.
Provider is selected by MEMORY_V3_EMBED_PROVIDER env var (ollama|openai|local).
"""

import hashlib
import json
from pathlib import Path

from .config import get_config
from .providers import get_embedder

CACHE_DIR = Path(get_config().cache_dir)


def embed_text(text: str) -> list[float]:
    """Embed a single text string using the configured provider. Returns float vector."""
    embedder = get_embedder()
    return embedder.embed(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured provider."""
    embedder = get_embedder()
    return embedder.embed_batch(texts)


def embed_with_cache(text: str) -> list[float]:
    """Embed with local file cache to avoid re-embedding identical content."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = get_config()
    cache_key = hashlib.sha256(
        f"{cfg.embed_provider}:{cfg.embed_dim}:{text}".encode()
    ).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    embedding = embed_text(text)
    cache_file.write_text(json.dumps(embedding))
    return embedding


def get_client():
    """Compatibility shim -- returns None; provider instances are created internally."""
    return None
