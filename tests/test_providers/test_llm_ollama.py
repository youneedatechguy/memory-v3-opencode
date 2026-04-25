from unittest.mock import patch, MagicMock
from memory_v3.providers.llm_ollama import OllamaLLM


def test_chat_returns_string():
    with patch("memory_v3.providers.llm_ollama.ollama") as mock_ollama:
        mock_ollama.chat.return_value = {"message": {"content": "extracted fact"}}
        llm = OllamaLLM(model="qwen2.5:3b")
        result = llm.chat("extract facts from: hello", max_tokens=512)
        assert result == "extracted fact"
        mock_ollama.chat.assert_called_once_with(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": "extract facts from: hello"}],
            options={"temperature": 0.1, "num_predict": 512},
        )


def test_chat_returns_empty_string_on_error():
    with patch("memory_v3.providers.llm_ollama.ollama") as mock_ollama:
        mock_ollama.chat.side_effect = Exception("connection refused")
        llm = OllamaLLM(model="qwen2.5:3b")
        result = llm.chat("prompt", max_tokens=256)
        assert result == ""
