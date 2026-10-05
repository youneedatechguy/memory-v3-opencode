import importlib
import sys
from types import SimpleNamespace


def _import_provider(monkeypatch, client):
    module_name = "memory_v3.providers.llm_openai"
    fake_openai = SimpleNamespace(OpenAI=lambda: client)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _completion(content):
    message = SimpleNamespace(content=f"  {content}  ")
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_chat_returns_string(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_MODEL", "invalid-ambient-model")
    completions = FakeCompletions(response=_completion("the answer"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    module = _import_provider(monkeypatch, client)

    llm = module.OpenAILLM(model="gpt-4o-mini")
    result = llm.chat("extract facts", max_tokens=512)

    assert result == "the answer"
    assert completions.calls == [
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "extract facts"}],
            "temperature": 0.1,
            "max_tokens": 512,
        }
    ]


def test_chat_returns_empty_string_on_error(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_MODEL", "invalid-ambient-model")
    completions = FakeCompletions(error=RuntimeError("rate limit"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    module = _import_provider(monkeypatch, client)

    llm = module.OpenAILLM(model="gpt-4o-mini")
    result = llm.chat("prompt", max_tokens=256)

    assert result == ""
    assert completions.calls == [
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "prompt"}],
            "temperature": 0.1,
            "max_tokens": 256,
        }
    ]
