from unittest.mock import patch, MagicMock
from memory_v3.providers.embedder_ollama import OllamaEmbedder


def test_embed_returns_vector():
    with patch("memory_v3.providers.embedder_ollama.ollama") as mock_ollama:
        mock_ollama.embed.return_value = MagicMock(embeddings=[[0.1] * 768])
        embedder = OllamaEmbedder(model="nomic-embed-text", dim=768)
        result = embedder.embed("hello world")
        assert len(result) == 768
        mock_ollama.embed.assert_called_once_with(model="nomic-embed-text", input="hello world")


def test_embed_batch_returns_matrix():
    with patch("memory_v3.providers.embedder_ollama.ollama") as mock_ollama:
        mock_ollama.embed.return_value = MagicMock(embeddings=[[0.1] * 768, [0.2] * 768])
        embedder = OllamaEmbedder(model="nomic-embed-text", dim=768)
        result = embedder.embed_batch(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 768
