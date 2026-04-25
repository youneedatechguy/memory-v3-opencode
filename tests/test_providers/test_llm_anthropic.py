from unittest.mock import patch, MagicMock
from memory_v3.providers.llm_anthropic import AnthropicLLM


def _mock_response(text):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def test_chat_returns_string():
    with patch("memory_v3.providers.llm_anthropic.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = _mock_response("extracted fact")
        llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
        result = llm.chat("extract facts", max_tokens=512)
        assert result == "extracted fact"
        MockClient.return_value.messages.create.assert_called_once_with(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            temperature=0.1,
            messages=[{"role": "user", "content": "extract facts"}],
        )


def test_chat_returns_empty_string_on_error():
    with patch("memory_v3.providers.llm_anthropic.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.side_effect = Exception("api error")
        llm = AnthropicLLM(model="claude-haiku-4-5-20251001")
        result = llm.chat("prompt", max_tokens=256)
        assert result == ""
