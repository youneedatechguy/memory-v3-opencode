import os
import pytest

sentence_transformers_available = False
try:
    import sentence_transformers
    sentence_transformers_available = True
except ImportError:
    pass


def test_get_embedder_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("MEMORY_V3_EMBED_PROVIDER", raising=False)
    from memory_v3.providers import get_embedder
    from memory_v3.providers.embedder_ollama import OllamaEmbedder
    assert isinstance(get_embedder(), OllamaEmbedder)


def test_get_embedder_openai(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "openai")
    from memory_v3.providers import get_embedder
    from memory_v3.providers.embedder_openai import OpenAIEmbedder
    assert isinstance(get_embedder(), OpenAIEmbedder)


@pytest.mark.skipif(not sentence_transformers_available, reason="sentence-transformers not installed")
def test_get_embedder_local(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "local")
    from memory_v3.providers import get_embedder
    from memory_v3.providers.embedder_local import LocalEmbedder
    assert isinstance(get_embedder(), LocalEmbedder)


def test_get_embedder_unknown_raises(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "banana")
    from memory_v3.providers import get_embedder
    with pytest.raises(ValueError, match="Unknown embedding provider 'banana'"):
        get_embedder()


def test_get_llm_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("MEMORY_V3_LLM_PROVIDER", raising=False)
    from memory_v3.providers import get_llm
    from memory_v3.providers.llm_ollama import OllamaLLM
    assert isinstance(get_llm(), OllamaLLM)


def test_get_llm_anthropic(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_PROVIDER", "anthropic")
    from memory_v3.providers import get_llm
    from memory_v3.providers.llm_anthropic import AnthropicLLM
    assert isinstance(get_llm(), AnthropicLLM)


def test_get_llm_openai(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_PROVIDER", "openai")
    from memory_v3.providers import get_llm
    from memory_v3.providers.llm_openai import OpenAILLM
    assert isinstance(get_llm(), OpenAILLM)


def test_get_llm_unknown_raises(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_PROVIDER", "banana")
    from memory_v3.providers import get_llm
    with pytest.raises(ValueError, match="Unknown LLM provider 'banana'"):
        get_llm()
