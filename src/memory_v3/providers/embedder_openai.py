"""OpenAI embedding provider."""
from __future__ import annotations

import os

try:
    import openai
except ImportError as e:
    raise ImportError(
        "openai package required for OpenAI embeddings. "
        "Install with: pip install 'memory-v3[openai]'"
    ) from e


class OpenAIEmbedder:
    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model = model or os.environ.get("MEMORY_V3_EMBED_MODEL", "text-embedding-3-small")
        self.dim = dim or int(os.environ.get("MEMORY_V3_EMBED_DIM", "1536"))
        self._client = openai.OpenAI()

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]
