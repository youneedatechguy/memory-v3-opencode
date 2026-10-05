from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from memory_v3.providers.embedder_ollama import OllamaEmbedder


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "invalid-ambient-provider")
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "invalid-ambient-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "999")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(
        "socket.create_connection", MagicMock(side_effect=AssertionError("network disabled"))
    )


def test_embed_returns_vector(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "explicit-ollama-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "4")
    with patch("memory_v3.providers.embedder_ollama.ollama") as mock_ollama:
        mock_ollama.embed.return_value = SimpleNamespace(embeddings=[[0.1] * 4])
        embedder = OllamaEmbedder(model="explicit-ollama-model", dim=4)
        result = embedder.embed("hello world")
        assert result == [0.1] * 4
        mock_ollama.embed.assert_called_once_with(
            model="explicit-ollama-model", input="hello world"
        )


def test_embed_batch_returns_matrix(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "explicit-ollama-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "4")
    with patch("memory_v3.providers.embedder_ollama.ollama") as mock_ollama:
        mock_ollama.embed.return_value = SimpleNamespace(embeddings=[[0.1] * 4, [0.2] * 4])
        embedder = OllamaEmbedder(model="explicit-ollama-model", dim=4)
        result = embedder.embed_batch(["text1", "text2"])
        assert result == [[0.1] * 4, [0.2] * 4]
        mock_ollama.embed.assert_called_once_with(
            model="explicit-ollama-model", input=["text1", "text2"]
        )


def test_uses_explicit_env_when_constructor_args_omitted(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "env-ollama-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "3")

    embedder = OllamaEmbedder()

    assert embedder.model == "env-ollama-model"
    assert embedder.dim == 3
