import importlib
import sys


def _import_provider(monkeypatch, ollama):
    module_name = "memory_v3.providers.llm_ollama"
    monkeypatch.setitem(sys.modules, "ollama", ollama)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    return importlib.import_module(module_name)


class FakeOllama:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_chat_returns_string(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_MODEL", "invalid-ambient-model")
    fake_ollama = FakeOllama(response={"message": {"content": "  extracted fact  "}})
    module = _import_provider(monkeypatch, fake_ollama)

    llm = module.OllamaLLM(model="qwen2.5:3b")
    result = llm.chat("extract facts from: hello", max_tokens=512)

    assert result == "extracted fact"
    assert fake_ollama.calls == [
        {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "extract facts from: hello"}],
            "options": {"temperature": 0.1, "num_predict": 512},
        }
    ]


def test_chat_returns_empty_string_on_error(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_MODEL", "invalid-ambient-model")
    fake_ollama = FakeOllama(error=RuntimeError("connection refused"))
    module = _import_provider(monkeypatch, fake_ollama)

    llm = module.OllamaLLM(model="qwen2.5:3b")
    result = llm.chat("prompt", max_tokens=256)

    assert result == ""
    assert fake_ollama.calls == [
        {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "prompt"}],
            "options": {"temperature": 0.1, "num_predict": 256},
        }
    ]
