from unittest.mock import patch, MagicMock
from memory_v3.providers.embedder_openai import OpenAIEmbedder


def _mock_response(vecs):
    resp = MagicMock()
    resp.data = [MagicMock(embedding=v) for v in vecs]
    return resp


def test_embed_returns_vector():
    with patch("memory_v3.providers.embedder_openai.openai.OpenAI") as MockClient:
        MockClient.return_value.embeddings.create.return_value = _mock_response([[0.5] * 1536])
        embedder = OpenAIEmbedder(model="text-embedding-3-small", dim=1536)
        result = embedder.embed("hello world")
        assert len(result) == 1536
        MockClient.return_value.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="hello world"
        )


def test_embed_batch_returns_matrix():
    with patch("memory_v3.providers.embedder_openai.openai.OpenAI") as MockClient:
        MockClient.return_value.embeddings.create.return_value = _mock_response(
            [[0.1] * 1536, [0.2] * 1536]
        )
        embedder = OpenAIEmbedder(model="text-embedding-3-small", dim=1536)
        result = embedder.embed_batch(["a", "b"])
        assert len(result) == 2
        assert len(result[0]) == 1536
