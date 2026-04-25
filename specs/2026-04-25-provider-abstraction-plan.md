# Provider Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hard-wired Ollama dependency with a pluggable provider system for embeddings (Ollama/OpenAI/local) and LLM calls (Ollama/OpenAI/Anthropic), then wire in OpenCode config and AGENTS.md.

**Architecture:** A `providers/` package exposes two factory functions (`get_embedder()`, `get_llm()`) that read `MEMORY_V3_EMBED_PROVIDER` / `MEMORY_V3_LLM_PROVIDER` env vars and return duck-typed provider instances. Three consumer files (`embeddings.py`, `extraction.py`, `compaction.py`) swap their direct `ollama.*` calls for the factory. A startup dim check in `server.py` catches dimension mismatches before accepting MCP calls. The `--reembed` CLI command performs intentional DB rebuilds.

**Tech Stack:** Python 3.10+, ollama>=0.4 (optional), openai>=1.0 (optional), anthropic>=0.40 (optional), sentence-transformers>=3.0 (optional), sqlite-vec, FastMCP

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/memory_v3/providers/__init__.py` | `get_embedder()` / `get_llm()` factories |
| Create | `src/memory_v3/providers/embedder_ollama.py` | Ollama embed wrapper |
| Create | `src/memory_v3/providers/embedder_openai.py` | OpenAI embed wrapper |
| Create | `src/memory_v3/providers/embedder_local.py` | sentence-transformers wrapper |
| Create | `src/memory_v3/providers/llm_ollama.py` | Ollama chat wrapper |
| Create | `src/memory_v3/providers/llm_openai.py` | OpenAI chat wrapper |
| Create | `src/memory_v3/providers/llm_anthropic.py` | Anthropic chat wrapper |
| Create | `tests/test_providers/` | Provider unit tests |
| Modify | `src/memory_v3/config.py` | Add `embed_provider`, `llm_provider` fields |
| Modify | `src/memory_v3/embeddings.py` | Use `get_embedder()`, fix cache key |
| Modify | `src/memory_v3/__init__.py` | Fix `get_embedder()` to return provider instance |
| Modify | `src/memory_v3/lifecycle/extraction.py` | Replace `ollama.chat()` with `get_llm().chat()` |
| Modify | `src/memory_v3/lifecycle/compaction.py` | Replace `ollama.chat()` with `get_llm().chat()` |
| Modify | `src/memory_v3/server.py` | Add `_check_embedding_dim()` at startup |
| Modify | `src/memory_v3/cli.py` | Add `--reembed` flag to reindex subcommand |
| Modify | `pyproject.toml` | Move ollama to optional, add openai/anthropic/local extras |
| Create | `AGENTS.md` | OpenCode project instructions |
| Create | `~/.config/opencode/config.json` | OpenCode MCP server config |

---

## Task 1: Provider package skeleton + factory tests

**Files:**
- Create: `tests/test_providers/__init__.py`
- Create: `tests/test_providers/test_factory.py`
- Create: `src/memory_v3/providers/__init__.py`

- [ ] **Step 1: Create test directory**

```bash
mkdir -p /home/base/memory-v3-opencode/tests/test_providers
touch /home/base/memory-v3-opencode/tests/test_providers/__init__.py
```

- [ ] **Step 2: Write failing factory tests**

Create `tests/test_providers/test_factory.py`:

```python
import os
import pytest


def test_get_embedder_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("MEMORY_V3_EMBED_PROVIDER", raising=False)
    from memory_v3.providers import get_embedder
    from memory_v3.providers.embedder_ollama import OllamaEmbedder
    assert isinstance(get_embedder(), OllamaEmbedder)


def test_get_embedder_openai(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "openai")
    from memory_v3.providers import get_embedder
    from memory_v3.providers.embedder_openai import OpenAIEmbedder
    assert isinstance(get_embedder(), OpenAIEmbedder)


def test_get_embedder_local(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "local")
    from memory_v3.providers import get_embedder
    from memory_v3.providers.embedder_local import LocalEmbedder
    assert isinstance(get_embedder(), LocalEmbedder)


def test_get_embedder_unknown_raises(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "banana")
    from memory_v3.providers import get_embedder
    with pytest.raises(ValueError, match="Unknown embedding provider 'banana'"):
        get_embedder()


def test_get_llm_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("MEMORY_V3_LLM_PROVIDER", raising=False)
    from memory_v3.providers import get_llm
    from memory_v3.providers.llm_ollama import OllamaLLM
    assert isinstance(get_llm(), OllamaLLM)


def test_get_llm_anthropic(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_PROVIDER", "anthropic")
    from memory_v3.providers import get_llm
    from memory_v3.providers.llm_anthropic import AnthropicLLM
    assert isinstance(get_llm(), AnthropicLLM)


def test_get_llm_openai(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_PROVIDER", "openai")
    from memory_v3.providers import get_llm
    from memory_v3.providers.llm_openai import OpenAILLM
    assert isinstance(get_llm(), OpenAILLM)


def test_get_llm_unknown_raises(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LLM_PROVIDER", "banana")
    from memory_v3.providers import get_llm
    with pytest.raises(ValueError, match="Unknown LLM provider 'banana'"):
        get_llm()
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd /home/base/memory-v3-opencode && pip install -e ".[dev]" -q && pytest tests/test_providers/test_factory.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'memory_v3.providers'`

- [ ] **Step 4: Create minimal providers package**

Create `src/memory_v3/providers/__init__.py`:

```python
"""Provider factory for embeddings and LLM calls."""
from __future__ import annotations

import os


def get_embedder():
    """Return the configured embedder instance."""
    provider = os.environ.get("MEMORY_V3_EMBED_PROVIDER", "ollama").lower()
    if provider == "ollama":
        from .embedder_ollama import OllamaEmbedder
        return OllamaEmbedder()
    if provider == "openai":
        from .embedder_openai import OpenAIEmbedder
        return OpenAIEmbedder()
    if provider == "local":
        from .embedder_local import LocalEmbedder
        return LocalEmbedder()
    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        "Valid options: ollama, openai, local"
    )


def get_llm():
    """Return the configured LLM instance."""
    provider = os.environ.get("MEMORY_V3_LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        from .llm_ollama import OllamaLLM
        return OllamaLLM()
    if provider == "openai":
        from .llm_openai import OpenAILLM
        return OpenAILLM()
    if provider == "anthropic":
        from .llm_anthropic import AnthropicLLM
        return AnthropicLLM()
    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Valid options: ollama, openai, anthropic"
    )
```

- [ ] **Step 5: Run tests — expect import errors for missing provider modules (that's correct progress)**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_factory.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'memory_v3.providers.embedder_ollama'` — confirms factory is found, stubs still missing.

---

## Task 2: OllamaEmbedder

**Files:**
- Create: `src/memory_v3/providers/embedder_ollama.py`
- Create: `tests/test_providers/test_embedder_ollama.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_providers/test_embedder_ollama.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_embedder_ollama.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'memory_v3.providers.embedder_ollama'`

- [ ] **Step 3: Implement OllamaEmbedder**

Create `src/memory_v3/providers/embedder_ollama.py`:

```python
"""Ollama embedding provider."""
from __future__ import annotations

import os

import ollama


class OllamaEmbedder:
    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
    ):
        self.model = model or os.environ.get("MEMORY_V3_EMBED_MODEL", "nomic-embed-text")
        self.dim = dim or int(os.environ.get("MEMORY_V3_EMBED_DIM", "768"))

    def embed(self, text: str) -> list[float]:
        resp = ollama.embed(model=self.model, input=text)
        return resp.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = ollama.embed(model=self.model, input=texts)
        return resp.embeddings
```

- [ ] **Step 4: Run tests**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_embedder_ollama.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/providers/ tests/test_providers/ && git commit -m "feat: add providers package with OllamaEmbedder and factory skeleton"
```

---

## Task 3: OpenAIEmbedder

**Files:**
- Create: `src/memory_v3/providers/embedder_openai.py`
- Create: `tests/test_providers/test_embedder_openai.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_providers/test_embedder_openai.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_embedder_openai.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'memory_v3.providers.embedder_openai'`

- [ ] **Step 3: Install openai package**

```bash
pip install openai>=1.0 -q
```

- [ ] **Step 4: Implement OpenAIEmbedder**

Create `src/memory_v3/providers/embedder_openai.py`:

```python
"""OpenAI embedding provider."""
from __future__ import annotations

import os

try:
    import openai
except ImportError as e:
    raise ImportError(
        "openai package required for OpenAI embeddings. "
        "Install with: pip install 'memory-v3[openai]'"
    ) from e


class OpenAIEmbedder:
    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
    ):
        self.model = model or os.environ.get("MEMORY_V3_EMBED_MODEL", "text-embedding-3-small")
        self.dim = dim or int(os.environ.get("MEMORY_V3_EMBED_DIM", "1536"))
        self._client = openai.OpenAI()

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]
```

- [ ] **Step 5: Run tests**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_embedder_openai.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/providers/embedder_openai.py tests/test_providers/test_embedder_openai.py && git commit -m "feat: add OpenAIEmbedder provider"
```

---

## Task 4: LocalEmbedder (sentence-transformers)

**Files:**
- Create: `src/memory_v3/providers/embedder_local.py`
- Create: `tests/test_providers/test_embedder_local.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_providers/test_embedder_local.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_embedder_local.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'memory_v3.providers.embedder_local'`

- [ ] **Step 3: Implement LocalEmbedder**

Create `src/memory_v3/providers/embedder_local.py`:

```python
"""Local sentence-transformers embedding provider. No API key required."""
from __future__ import annotations

import os

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers package required for local embeddings. "
        "Install with: pip install 'memory-v3[local]'"
    ) from e


class LocalEmbedder:
    """Embedding provider using sentence-transformers (runs on CPU, no API key).

    Recommended model: all-mpnet-base-v2 (dim=768, compatible with Ollama nomic-embed-text).
    Set MEMORY_V3_EMBED_MODEL and MEMORY_V3_EMBED_DIM to match your chosen model.
    """

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
    ):
        self.model_name = model or os.environ.get("MEMORY_V3_EMBED_MODEL", "all-mpnet-base-v2")
        self.dim = dim or int(os.environ.get("MEMORY_V3_EMBED_DIM", "768"))
        self._model = SentenceTransformer(self.model_name)

        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self.dim:
            print(
                f"WARNING: LocalEmbedder loaded '{self.model_name}' which outputs dim={actual_dim}, "
                f"but MEMORY_V3_EMBED_DIM={self.dim}. "
                f"Set MEMORY_V3_EMBED_DIM={actual_dim} to match, or the startup dim check will block."
            )

    def embed(self, text: str) -> list[float]:
        vecs = self._model.encode([text], convert_to_numpy=True)
        return vecs[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vecs]
```

- [ ] **Step 4: Run tests**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_embedder_local.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/providers/embedder_local.py tests/test_providers/test_embedder_local.py && git commit -m "feat: add LocalEmbedder (sentence-transformers) provider"
```

---

## Task 5: OllamaLLM

**Files:**
- Create: `src/memory_v3/providers/llm_ollama.py`
- Create: `tests/test_providers/test_llm_ollama.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_providers/test_llm_ollama.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_llm_ollama.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'memory_v3.providers.llm_ollama'`

- [ ] **Step 3: Implement OllamaLLM**

Create `src/memory_v3/providers/llm_ollama.py`:

```python
"""Ollama LLM provider."""
from __future__ import annotations

import os

import ollama


class OllamaLLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MEMORY_V3_LLM_MODEL", "qwen2.5:3b")

    def chat(self, prompt: str, max_tokens: int = 2048) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": max_tokens},
            )
            return response["message"]["content"].strip()
        except Exception:
            return ""
```

- [ ] **Step 4: Run tests**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_llm_ollama.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/providers/llm_ollama.py tests/test_providers/test_llm_ollama.py && git commit -m "feat: add OllamaLLM provider"
```

---

## Task 6: OpenAILLM

**Files:**
- Create: `src/memory_v3/providers/llm_openai.py`
- Create: `tests/test_providers/test_llm_openai.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_providers/test_llm_openai.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_llm_openai.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'memory_v3.providers.llm_openai'`

- [ ] **Step 3: Implement OpenAILLM**

Create `src/memory_v3/providers/llm_openai.py`:

```python
"""OpenAI LLM provider."""
from __future__ import annotations

import os

try:
    import openai
except ImportError as e:
    raise ImportError(
        "openai package required. Install with: pip install 'memory-v3[openai]'"
    ) from e


class OpenAILLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MEMORY_V3_LLM_MODEL", "gpt-4o-mini")
        self._client = openai.OpenAI()

    def chat(self, prompt: str, max_tokens: int = 2048) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""
```

- [ ] **Step 4: Run tests**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_llm_openai.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/providers/llm_openai.py tests/test_providers/test_llm_openai.py && git commit -m "feat: add OpenAILLM provider"
```

---

## Task 7: AnthropicLLM

**Files:**
- Create: `src/memory_v3/providers/llm_anthropic.py`
- Create: `tests/test_providers/test_llm_anthropic.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_providers/test_llm_anthropic.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_llm_anthropic.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'memory_v3.providers.llm_anthropic'`

- [ ] **Step 3: Install anthropic package**

```bash
pip install anthropic>=0.40 -q
```

- [ ] **Step 4: Implement AnthropicLLM**

Create `src/memory_v3/providers/llm_anthropic.py`:

```python
"""Anthropic LLM provider."""
from __future__ import annotations

import os

try:
    import anthropic
except ImportError as e:
    raise ImportError(
        "anthropic package required. Install with: pip install 'memory-v3[anthropic]'"
    ) from e


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MEMORY_V3_LLM_MODEL", "claude-haiku-4-5-20251001")
        self._client = anthropic.Anthropic()

    def chat(self, prompt: str, max_tokens: int = 2048) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception:
            return ""
```

- [ ] **Step 5: Run tests**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/test_llm_anthropic.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Run all provider tests together**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_providers/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/providers/llm_anthropic.py tests/test_providers/test_llm_anthropic.py && git commit -m "feat: add AnthropicLLM provider — all provider tests passing"
```

---

## Task 8: Update config.py

**Files:**
- Modify: `src/memory_v3/config.py:160-200` (Config dataclass and _build_config)

- [ ] **Step 1: Add fields to Config dataclass**

In `src/memory_v3/config.py`, add two fields to the `Config` dataclass after the `embed_dim` field (around line 56):

```python
    # --- Provider selection ---
    embed_provider: str = "ollama"   # MEMORY_V3_EMBED_PROVIDER
    llm_provider: str = "ollama"     # MEMORY_V3_LLM_PROVIDER
```

- [ ] **Step 2: Wire fields in _build_config()**

In `_build_config()`, add after the `embed_dim` line (around line 175):

```python
        embed_provider=_env("EMBED_PROVIDER", "ollama"),
        llm_provider=_env("LLM_PROVIDER", "ollama"),
```

- [ ] **Step 3: Verify config loads cleanly**

```bash
cd /home/base/memory-v3-opencode && python -c "
from memory_v3.config import get_config, reset_config
import os
os.environ['MEMORY_V3_EMBED_PROVIDER'] = 'openai'
os.environ['MEMORY_V3_LLM_PROVIDER'] = 'anthropic'
reset_config()
cfg = get_config()
assert cfg.embed_provider == 'openai'
assert cfg.llm_provider == 'anthropic'
print('config OK')
"
```

Expected: `config OK`

- [ ] **Step 4: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/config.py && git commit -m "feat: add embed_provider and llm_provider to Config"
```

---

## Task 9: Update embeddings.py

**Files:**
- Modify: `src/memory_v3/embeddings.py`

- [ ] **Step 1: Rewrite embeddings.py**

Replace the entire contents of `src/memory_v3/embeddings.py`:

```python
"""
memory-v3 -- Embedding Layer
Provider-agnostic. Configured via MEMORY_V3_EMBED_PROVIDER env var.
Providers: ollama (default), openai, local (sentence-transformers).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get(
    "MEMORY_V3_CACHE",
    str(Path.home() / ".memory-v3" / "cache")
))

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from .providers import get_embedder
        _embedder = get_embedder()
    return _embedder


def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns float vector."""
    return _get_embedder().embed(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts."""
    return _get_embedder().embed_batch(texts)


def embed_with_cache(text: str) -> list[float]:
    """Embed with local file cache. Cache key includes provider and dim."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    provider = os.environ.get("MEMORY_V3_EMBED_PROVIDER", "ollama")
    dim = os.environ.get("MEMORY_V3_EMBED_DIM", "768")
    cache_key = hashlib.sha256(f"{provider}:{dim}:{text}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    embedding = embed_text(text)
    cache_file.write_text(json.dumps(embedding))
    return embedding


def get_client():
    """Compatibility shim — kept for any callers that reference it."""
    return None
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
cd /home/base/memory-v3-opencode && python -c "from memory_v3.embeddings import embed_text, embed_batch, embed_with_cache; print('embeddings OK')"
```

Expected: `embeddings OK`

- [ ] **Step 3: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/embeddings.py && git commit -m "feat: wire embeddings.py to provider factory, fix cache key"
```

---

## Task 10: Update __init__.py get_embedder()

**Files:**
- Modify: `src/memory_v3/__init__.py`

- [ ] **Step 1: Replace get_embedder() implementation**

Replace the entire `__init__.py`:

```python
"""memory-v3: Next-generation brain-inspired persistent memory for AI coding assistants."""

__version__ = "1.0.0"
__author__ = "Sean Pembroke"


def get_embedder():
    """Return the configured embedder instance (lazy-loaded)."""
    from .providers import get_embedder as _get_embedder
    return _get_embedder()


def get_version_info() -> dict:
    """Return version metadata for diagnostics."""
    return {
        "name": "memory-v3",
        "version": __version__,
        "author": __author__,
        "description": "Brain-inspired persistent memory with governance layers, "
                       "sensory gating, and multi-graph knowledge representation.",
    }
```

- [ ] **Step 2: Verify server.py import still works**

```bash
cd /home/base/memory-v3-opencode && python -c "from memory_v3 import get_embedder, get_version_info; print(get_version_info()['name'])"
```

Expected: `memory-v3`

- [ ] **Step 3: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/__init__.py && git commit -m "refactor: get_embedder() in __init__ delegates to provider factory"
```

---

## Task 11: Update extraction.py

**Files:**
- Modify: `src/memory_v3/lifecycle/extraction.py`

- [ ] **Step 1: Remove `import ollama` and replace the ollama.chat() call**

In `extraction.py`, make two changes:

**Change 1** — remove `import ollama` from the imports (around line 20).

**Change 2** — replace the `ollama.chat()` block in `extract_facts()` (lines 169-175):

Old:
```python
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 2048},
        )
        content = response["message"]["content"].strip()
    except Exception as e:
        return [f"[extraction error: {e}]"]
```

New:
```python
    try:
        from ..providers import get_llm
        content = get_llm().chat(prompt, max_tokens=2048)
    except Exception as e:
        return [f"[extraction error: {e}]"]
```

Also remove the `model = cfg.llm_model` line from `extract_facts()` since the provider reads it from env directly. The function signature and all other logic stays identical.

- [ ] **Step 2: Verify extraction module imports cleanly**

```bash
cd /home/base/memory-v3-opencode && python -c "from memory_v3.lifecycle.extraction import extract_facts, process_conversation; print('extraction OK')"
```

Expected: `extraction OK`

- [ ] **Step 3: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/lifecycle/extraction.py && git commit -m "refactor: extraction.py uses get_llm() provider, removes ollama import"
```

---

## Task 12: Update compaction.py

**Files:**
- Modify: `src/memory_v3/lifecycle/compaction.py`

- [ ] **Step 1: Remove `import ollama` and replace the ollama.chat() call**

In `compaction.py`, make two changes:

**Change 1** — remove `import ollama` from the imports (around line 23).

**Change 2** — replace the `ollama.chat()` block in `structured_summary()` (lines 149-156):

Old:
```python
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": max(256, target_len * 2)},
        )
        summary = response["message"]["content"].strip()
        return summary
    except Exception as e:
```

New:
```python
    try:
        from ..providers import get_llm
        summary = get_llm().chat(prompt, max_tokens=max(256, target_len * 2))
        return summary
    except Exception as e:
```

Also remove `model = cfg.llm_model` from `structured_summary()`.

- [ ] **Step 2: Verify compaction module imports cleanly**

```bash
cd /home/base/memory-v3-opencode && python -c "from memory_v3.lifecycle.compaction import compact, structured_summary; print('compaction OK')"
```

Expected: `compaction OK`

- [ ] **Step 3: Run full test suite to catch any regressions**

```bash
cd /home/base/memory-v3-opencode && pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all existing tests pass (provider tests + original tests).

- [ ] **Step 4: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/lifecycle/compaction.py && git commit -m "refactor: compaction.py uses get_llm() provider, removes ollama import"
```

---

## Task 13: Startup dim check in server.py

**Files:**
- Modify: `src/memory_v3/server.py`
- Create: `tests/test_dim_check.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_dim_check.py`:

```python
import sqlite3
import pytest
from unittest.mock import patch
from memory_v3.server import _check_embedding_dim


def _make_conn_with_dim(dim: int):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sqlite_master_mock (sql TEXT)")
    return conn, f"CREATE VIRTUAL TABLE memory_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[{dim}])"


def test_matching_dim_passes(tmp_path):
    import sqlite3, sqlite_vec
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("""CREATE VIRTUAL TABLE memory_vec USING vec0(
        id INTEGER PRIMARY KEY, embedding float[768])""")
    _check_embedding_dim(conn, 768)  # should not raise or exit


def test_mismatched_dim_exits(tmp_path, capsys):
    import sqlite3, sqlite_vec
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("""CREATE VIRTUAL TABLE memory_vec USING vec0(
        id INTEGER PRIMARY KEY, embedding float[768])""")
    with pytest.raises(SystemExit) as exc_info:
        _check_embedding_dim(conn, 1536)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "mismatch" in captured.out.lower()
    assert "cost" in captured.out.lower()
    assert "memory-v3-reindex --reembed" in captured.out
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_dim_check.py -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name '_check_embedding_dim' from 'memory_v3.server'`

- [ ] **Step 3: Add _check_embedding_dim to server.py**

Add this function to `server.py` after the imports, before `mcp = FastMCP(...)`:

```python
def _check_embedding_dim(conn, configured_dim: int) -> None:
    """Check memory_vec schema dim matches configured dim. Exit with message if not."""
    import re
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_vec'"
    ).fetchone()
    if row is None:
        return  # DB not initialised yet — skip check
    sql = row[0]
    match = re.search(r"float\[(\d+)\]", sql)
    if match is None:
        return  # Can't parse — skip check
    db_dim = int(match.group(1))
    if db_dim == configured_dim:
        return
    print(
        f"\nERROR: Embedding dimension mismatch.\n"
        f"  DB was created with dim={db_dim} (e.g. nomic-embed-text / Ollama).\n"
        f"  Current config requests dim={configured_dim} (e.g. text-embedding-3-small / OpenAI).\n"
        f"\n"
        f"This means all existing memories need to be re-embedded before the server can run.\n"
        f"Re-embedding sends every memory through your new embedding provider's API,\n"
        f"which may incur significant cost (e.g. ~$0.02 per 1M tokens for OpenAI —\n"
        f"costs add up quickly with thousands of memories).\n"
        f"\n"
        f"To proceed intentionally, run:\n"
        f"  memory-v3-reindex --reembed\n"
        f"\n"
        f"This will flush the embedding cache, rebuild memory_vec at dim={configured_dim},\n"
        f"and recompute all cluster centroids. Your memories are NOT deleted.\n"
    )
    import sys
    sys.exit(1)
```

Then call it inside `run()` before `mcp.run()`:

```python
def run():
    """Entry point for the MCP server."""
    from .db import get_connection
    from .config import get_config
    cfg = get_config()
    conn = get_connection(cfg.db_path)
    _check_embedding_dim(conn, cfg.embed_dim)
    mcp.run(show_banner=False)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/base/memory-v3-opencode && pytest tests/test_dim_check.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/server.py tests/test_dim_check.py && git commit -m "feat: add startup embedding dim check with cost warning"
```

---

## Task 14: Add --reembed to CLI

**Files:**
- Modify: `src/memory_v3/cli.py`

- [ ] **Step 1: Add --reembed argument to reindex subparser**

In `build_parser()`, find the reindex subparser block (around line 315) and add `--reembed`:

```python
    # --- reindex ---
    p = sub.add_parser("reindex", help="Re-index the vault")
    p.add_argument("--force", action="store_true", help="Force full re-index")
    p.add_argument(
        "--reembed",
        action="store_true",
        help="Drop and rebuild memory_vec with the current embedding provider. "
             "Prompts for confirmation before running (may incur API costs).",
    )
```

- [ ] **Step 2: Add reembed logic to cmd_reindex()**

Replace the existing `cmd_reindex()` function:

```python
def cmd_reindex(args):
    """Re-index the vault, with optional full re-embedding."""
    if getattr(args, "reembed", False):
        _cmd_reembed()
        return
    from .server import reindex
    result = reindex(force=args.force)
    _pp(result)


def _cmd_reembed():
    """Drop and rebuild memory_vec with the current embedding provider."""
    import sqlite3
    import sqlite_vec
    from .config import get_config
    from .db import get_connection
    from .embeddings import embed_text

    cfg = get_config()
    provider = cfg.embed_provider
    dim = cfg.embed_dim
    model = cfg.embed_model

    conn = get_connection(cfg.db_path)
    count = conn.execute("SELECT COUNT(*) FROM memories WHERE archived = 0").fetchone()[0]

    print(
        f"\nRe-embed Summary\n"
        f"  Provider : {provider}\n"
        f"  Model    : {model}\n"
        f"  Dim      : {dim}\n"
        f"  Memories : {count} (archived memories excluded)\n"
        f"\n"
        f"WARNING: This will send {count} texts to your embedding provider.\n"
        f"If using a paid API (OpenAI, etc.) this may incur charges.\n"
        f"Estimated tokens depend on your memory content lengths.\n"
    )
    answer = input(f"Re-embed {count} memories using '{provider}'? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    import json
    import hashlib
    from pathlib import Path
    from .db import _serialize_f32

    # Flush embedding cache
    cache_dir = Path(cfg.cache_dir)
    if cache_dir.exists():
        removed = 0
        for f in cache_dir.glob("*.json"):
            f.unlink()
            removed += 1
        print(f"Flushed {removed} cached embeddings.")

    # Drop and recreate memory_vec
    conn.execute("DROP TABLE IF EXISTS memory_vec")
    conn.execute(f"""
        CREATE VIRTUAL TABLE memory_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{dim}]
        )
    """)
    conn.commit()
    print(f"Rebuilt memory_vec at float[{dim}].")

    # Re-embed all active memories in batches of 50
    rows = conn.execute(
        "SELECT id, content FROM memories WHERE archived = 0 ORDER BY id"
    ).fetchall()

    errors = 0
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        try:
            from .embeddings import embed_batch
            vecs = embed_batch(texts)
            for mid, vec in zip(ids, vecs):
                conn.execute(
                    "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                    (mid, _serialize_f32(vec)),
                )
            conn.commit()
            print(f"  Re-embedded {min(i + batch_size, len(rows))}/{len(rows)}...")
        except Exception as e:
            errors += 1
            print(f"  ERROR on batch starting at id={ids[0]}: {e}")

    # Recompute cluster centroids
    try:
        from .graphs.communities import recompute_centroids
        recompute_centroids(conn)
        print("Recomputed cluster centroids.")
    except Exception as e:
        print(f"Centroid recomputation failed (non-fatal): {e}")

    print(
        f"\nDone. {len(rows) - errors * batch_size}/{len(rows)} memories re-embedded. "
        f"Errors: {errors} batches."
    )
```

- [ ] **Step 3: Verify CLI parses the new flag**

```bash
cd /home/base/memory-v3-opencode && python -m memory_v3.cli reindex --help 2>&1 | grep reembed
```

Expected: `--reembed   Drop and rebuild memory_vec...`

- [ ] **Step 4: Commit**

```bash
cd /home/base/memory-v3-opencode && git add src/memory_v3/cli.py && git commit -m "feat: add memory-v3-reindex --reembed command with cost warning"
```

---

## Task 15: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Move ollama to optional, add new provider extras**

Replace the `[project.optional-dependencies]` section and `dependencies` in `pyproject.toml`:

```toml
dependencies = [
    "fastmcp>=2.0",
    "numpy>=1.24",
    "networkx>=3.0",
    "sqlite-vec>=0.1",
]

[project.optional-dependencies]
ollama = ["ollama>=0.4"]
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.40"]
local = ["sentence-transformers>=3.0"]
graph = ["python-igraph>=0.11", "leidenalg>=0.10", "pyvis>=0.3"]
dev = ["pytest>=8.0", "pytest-cov", "ruff"]
all = ["memory-v3[ollama,openai,anthropic,local,graph,dev]"]
```

- [ ] **Step 2: Verify package installs without ollama as a hard dep**

```bash
cd /home/base/memory-v3-opencode && pip install -e "." -q && python -c "import memory_v3; print('core install OK, no ollama required')"
```

Expected: `core install OK, no ollama required`

- [ ] **Step 3: Verify [all] still works**

```bash
cd /home/base/memory-v3-opencode && pip install -e ".[all]" -q && python -c "import memory_v3; print('all extras OK')"
```

Expected: `all extras OK`

- [ ] **Step 4: Run full test suite one final time**

```bash
cd /home/base/memory-v3-opencode && pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/base/memory-v3-opencode && git add pyproject.toml && git commit -m "build: move ollama to optional dep, add openai/anthropic/local extras"
```

---

## Task 16: AGENTS.md and OpenCode config

**Files:**
- Create: `AGENTS.md`
- Create: `~/.config/opencode/config.json`

- [ ] **Step 1: Create AGENTS.md**

Create `AGENTS.md` in the project root (copy of CLAUDE.md with OpenCode-specific session section prepended):

```bash
cp /home/base/memory-v3-opencode/CLAUDE.md /home/base/memory-v3-opencode/AGENTS.md
```

Then prepend the following block at the top of `AGENTS.md` (above the `# memory-v3-opencode` heading):

```markdown
## OpenCode Session Protocol

**At session start:**
1. `agent_sync("opencode")` — catch up on memory changes from other agents
2. `search("current project context")` — recall relevant memories
3. `list_recent` (last 24h) — if returning to in-progress work

**At session end or task switch:**
1. `extract_from_conversation` — pass a summary of decisions/changes made this session

---

```

- [ ] **Step 2: Create OpenCode config directory and config file**

```bash
mkdir -p ~/.config/opencode
```

Create `~/.config/opencode/config.json`:

```json
{
  "model": "anthropic/claude-sonnet-4-6",
  "mcpServers": {
    "memory-v3": {
      "type": "stdio",
      "command": "memory-v3-server",
      "env": {
        "MEMORY_V3_DB": "/home/base/.memory-v3/memory.db",
        "MEMORY_V3_EMBED_PROVIDER": "openai",
        "MEMORY_V3_LLM_PROVIDER": "anthropic",
        "MEMORY_V3_EMBED_MODEL": "text-embedding-3-small",
        "MEMORY_V3_EMBED_DIM": "1536",
        "MEMORY_V3_LLM_MODEL": "claude-haiku-4-5-20251001",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/base/memory-v3-opencode && git add AGENTS.md && git commit -m "docs: add AGENTS.md for OpenCode session protocol"
```

(The OpenCode config in `~/.config` is not committed — it's a user-local file.)

---

## Final Verification

- [ ] **Run complete test suite**

```bash
cd /home/base/memory-v3-opencode && pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Verify no remaining `import ollama` in lifecycle files**

```bash
grep -rn "import ollama" /home/base/memory-v3-opencode/src/memory_v3/lifecycle/
```

Expected: no output.

- [ ] **Verify `import ollama` only appears in the Ollama provider files**

```bash
grep -rn "import ollama" /home/base/memory-v3-opencode/src/
```

Expected: only `providers/embedder_ollama.py` and `providers/llm_ollama.py`.

- [ ] **Smoke test server startup (with Ollama provider, no DB)**

```bash
MEMORY_V3_EMBED_PROVIDER=ollama MEMORY_V3_LLM_PROVIDER=ollama timeout 3 memory-v3-server 2>&1 || true
```

Expected: server starts without ImportError (may hang waiting for MCP connection — that's fine, timeout kills it).
