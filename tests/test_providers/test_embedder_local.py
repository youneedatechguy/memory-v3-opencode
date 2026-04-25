from unittest.mock import patch, MagicMock
from memory_v3.providers.embedder_local import LocalEmbedder


def test_embed_returns_vector():
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.3] * 768]
    mock_model.get_sentence_embedding_dimension.return_value = 768
    with patch("memory_v3.providers.embedder_local.SentenceTransformer", return_value=mock_model):
        embedder = LocalEmbedder(model="all-mpnet-base-v2", dim=768)
        result = embedder.embed("hello world")
        assert len(result) == 768
        mock_model.encode.assert_called_once_with(["hello world"], convert_to_numpy=True)


def test_embed_batch_returns_matrix():
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]
    mock_model.get_sentence_embedding_dimension.return_value = 768
    with patch("memory_v3.providers.embedder_local.SentenceTransformer", return_value=mock_model):
        embedder = LocalEmbedder(model="all-mpnet-base-v2", dim=768)
        result = embedder.embed_batch(["a", "b"])
        assert len(result) == 2
        assert len(result[0]) == 768


def test_dim_mismatch_warns(capsys):
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 384]
    mock_model.get_sentence_embedding_dimension.return_value = 384
    with patch("memory_v3.providers.embedder_local.SentenceTransformer", return_value=mock_model):
        LocalEmbedder(model="all-MiniLM-L6-v2", dim=768)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "384" in captured.out
