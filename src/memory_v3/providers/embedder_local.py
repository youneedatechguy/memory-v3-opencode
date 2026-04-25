"""Local sentence-transformers embedding provider. No API key required."""
from __future__ import annotations

import os

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]


class LocalEmbedder:
    """Embedding provider using sentence-transformers (runs on CPU, no API key).

    Recommended model: all-mpnet-base-v2 (dim=768, compatible with Ollama nomic-embed-text).
    Set MEMORY_V3_EMBED_MODEL and MEMORY_V3_EMBED_DIM to match your chosen model.
    """

    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model_name = model or os.environ.get("MEMORY_V3_EMBED_MODEL", "all-mpnet-base-v2")
        self.dim = dim or int(os.environ.get("MEMORY_V3_EMBED_DIM", "768"))
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers package required for local embeddings. "
                "Install with: pip install 'memory-v3[local]'"
            )
        self._model = SentenceTransformer(self.model_name)

        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self.dim:
            print(
                f"WARNING: LocalEmbedder loaded '{self.model_name}' which outputs dim={actual_dim}, "
                f"but MEMORY_V3_EMBED_DIM={self.dim}. "
                f"Set MEMORY_V3_EMBED_DIM={actual_dim} to match, or the startup dim check will block."
            )

    def embed(self, text: str) -> list[float]:
        vecs = self._model.encode([text], convert_to_numpy=True)
        v = vecs[0]
        return v.tolist() if hasattr(v, "tolist") else list(v)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
