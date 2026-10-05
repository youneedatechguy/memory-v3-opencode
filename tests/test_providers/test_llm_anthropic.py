import importlib
import sys
from types import SimpleNamespace


def _import_provider(monkeypatch, client):
    module_name = "memory_v3.providers.llm_anthropic"
    fake_anthropic = SimpleNamespace(Anthropic=lambda: client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _response(text):
    return SimpleNamespace(content=[SimpleNamespace(text=f"  {text}  ")])


def test_chat_returns_string(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_MODEL", "invalid-ambient-model")
    messages = FakeMessages(response=_response("extracted fact"))
    module = _import_provider(monkeypatch, SimpleNamespace(messages=messages))

    llm = module.AnthropicLLM(model="claude-haiku-4-5-20251001")
    result = llm.chat("extract facts", max_tokens=512)

    assert result == "extracted fact"
    assert messages.calls == [
        {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 512,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": "extract facts"}],
        }
    ]


def test_chat_returns_empty_string_on_error(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_MODEL", "invalid-ambient-model")
    messages = FakeMessages(error=RuntimeError("api error"))
    module = _import_provider(monkeypatch, SimpleNamespace(messages=messages))

    llm = module.AnthropicLLM(model="claude-haiku-4-5-20251001")
    result = llm.chat("prompt", max_tokens=256)

    assert result == ""
    assert messages.calls == [
        {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": "prompt"}],
        }
    ]
