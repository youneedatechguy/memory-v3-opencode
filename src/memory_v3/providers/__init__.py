"""Provider factory for embeddings and LLM calls."""
from __future__ import annotations

import os


def get_embedder():
    """Return the configured embedder instance."""
    provider = os.environ.get("MEMORY_V3_EMBED_PROVIDER", "ollama").lower()
    if provider == "ollama":
        from .embedder_ollama import OllamaEmbedder
        return OllamaEmbedder()
    if provider == "openai":
        from .embedder_openai import OpenAIEmbedder
        return OpenAIEmbedder()
    if provider == "local":
        from .embedder_local import LocalEmbedder
        return LocalEmbedder()
    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        "Valid options: ollama, openai, local"
    )


def get_llm():
    """Return the configured LLM instance."""
    provider = os.environ.get("MEMORY_V3_LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        from .llm_ollama import OllamaLLM
        return OllamaLLM()
    if provider == "openai":
        from .llm_openai import OpenAILLM
        return OpenAILLM()
    if provider == "anthropic":
        from .llm_anthropic import AnthropicLLM
        return AnthropicLLM()
    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Valid options: ollama, openai, anthropic"
    )
