from unittest.mock import patch, MagicMock
from memory_v3.providers.llm_openai import OpenAILLM


def _mock_completion(content):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def test_chat_returns_string():
    with patch("memory_v3.providers.llm_openai.openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.return_value = _mock_completion("the answer")
        llm = OpenAILLM(model="gpt-4o-mini")
        result = llm.chat("extract facts", max_tokens=512)
        assert result == "the answer"
        MockClient.return_value.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "extract facts"}],
            temperature=0.1,
            max_tokens=512,
        )


def test_chat_returns_empty_string_on_error():
    with patch("memory_v3.providers.llm_openai.openai.OpenAI") as MockClient:
        MockClient.return_value.chat.completions.create.side_effect = Exception("rate limit")
        llm = OpenAILLM(model="gpt-4o-mini")
        result = llm.chat("prompt", max_tokens=256)
        assert result == ""
