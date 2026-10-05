from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import memory_v3.providers.embedder_openai as embedder_openai
from memory_v3.providers.embedder_openai import OpenAIEmbedder


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "invalid-ambient-provider")
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "invalid-ambient-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "999")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(
        "socket.create_connection", MagicMock(side_effect=AssertionError("network disabled"))
    )


def _mock_response(vecs):
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vecs])


def _fake_openai(response, calls):
    class FakeEmbeddings:
        def create(self, *, model, input):
            calls.append((model, input))
            return response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.embeddings = FakeEmbeddings()

    return SimpleNamespace(OpenAI=FakeOpenAI)


def test_embed_returns_vector(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "explicit-openai-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "4")
    calls = []
    monkeypatch.setattr(embedder_openai, "openai", _fake_openai(_mock_response([[0.5] * 4]), calls))

    embedder = OpenAIEmbedder(model="explicit-openai-model", dim=4)
    result = embedder.embed("hello world")

    assert result == [0.5] * 4
    assert calls == [{"max_retries": 0}, ("explicit-openai-model", "hello world")]


def test_embed_batch_returns_matrix(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "explicit-openai-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "4")
    calls = []
    monkeypatch.setattr(
        embedder_openai,
        "openai",
        _fake_openai(_mock_response([[0.1] * 4, [0.2] * 4]), calls),
    )

    embedder = OpenAIEmbedder(model="explicit-openai-model", dim=4)
    result = embedder.embed_batch(["a", "b"])

    assert result == [[0.1] * 4, [0.2] * 4]
    assert calls == [{"max_retries": 0}, ("explicit-openai-model", ["a", "b"])]


def test_uses_explicit_env_when_constructor_args_omitted(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "env-openai-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "3")

    embedder = OpenAIEmbedder()

    assert embedder.model == "env-openai-model"
    assert embedder.dim == 3
