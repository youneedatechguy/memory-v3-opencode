"""Ollama embedding provider."""
from __future__ import annotations

import os

import ollama


class OllamaEmbedder:
    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model = model or os.environ.get("MEMORY_V3_EMBED_MODEL", "nomic-embed-text")
        self.dim = dim or int(os.environ.get("MEMORY_V3_EMBED_DIM", "768"))

    def embed(self, text: str) -> list[float]:
        resp = ollama.embed(model=self.model, input=text)
        return resp.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = ollama.embed(model=self.model, input=texts)
        return resp.embeddings
