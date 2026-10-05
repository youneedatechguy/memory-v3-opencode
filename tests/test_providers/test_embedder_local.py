from unittest.mock import MagicMock, patch

import pytest

from memory_v3.providers.embedder_local import LocalEmbedder


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "invalid-ambient-provider")
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "invalid-ambient-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "999")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", "/does/not/exist")
    monkeypatch.setattr(
        "socket.create_connection", MagicMock(side_effect=AssertionError("network disabled"))
    )


def test_embed_returns_vector(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "explicit-local-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "4")
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.3] * 4]
    mock_model.get_embedding_dimension.return_value = 4
    with patch("memory_v3.providers.embedder_local.SentenceTransformer", return_value=mock_model):
        embedder = LocalEmbedder(model="explicit-local-model", dim=4)
        result = embedder.embed("hello world")
        assert result == [0.3] * 4
        mock_model.encode.assert_called_once_with(["hello world"], convert_to_numpy=True)


def test_embed_batch_returns_matrix(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "explicit-local-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "4")
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 4, [0.2] * 4]
    mock_model.get_embedding_dimension.return_value = 4
    with patch("memory_v3.providers.embedder_local.SentenceTransformer", return_value=mock_model):
        embedder = LocalEmbedder(model="explicit-local-model", dim=4)
        result = embedder.embed_batch(["a", "b"])
        assert result == [[0.1] * 4, [0.2] * 4]
        mock_model.encode.assert_called_once_with(["a", "b"], convert_to_numpy=True)


def test_dim_mismatch_warns(monkeypatch, capsys):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "explicit-local-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "7")
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 5]
    mock_model.get_embedding_dimension.return_value = 5
    with patch("memory_v3.providers.embedder_local.SentenceTransformer", return_value=mock_model):
        LocalEmbedder(model="explicit-local-model", dim=7)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "5" in captured.out
        assert "7" in captured.out


def test_uses_explicit_env_when_constructor_args_omitted(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "env-local-model")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", "3")
    mock_model = MagicMock()
    mock_model.get_embedding_dimension.return_value = 3

    with patch(
        "memory_v3.providers.embedder_local.SentenceTransformer", return_value=mock_model
    ) as loader:
        embedder = LocalEmbedder()

    assert embedder.model_name == "env-local-model"
    assert embedder.dim == 3
    loader.assert_called_once_with("env-local-model")
