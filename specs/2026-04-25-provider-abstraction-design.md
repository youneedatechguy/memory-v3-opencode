# Provider Abstraction Design

**Date:** 2026-04-25
**Repo:** memory-v3-opencode (fork of Haustorium12/memory-v3)
**Goal:** Replace hard Ollama dependency with a pluggable provider system for embeddings and LLM calls.

---

## Problem

The upstream codebase hardwires `ollama` in three places:
- `src/memory_v3/embeddings.py` — `ollama.embed()`
- `src/memory_v3/lifecycle/extraction.py` — `ollama.chat()`
- `src/memory_v3/lifecycle/compaction.py` — `ollama.chat()`

`ollama>=0.4` is a core dependency, so Ollama must be running locally even if the user wants to use OpenAI or Anthropic. The fork exists to fix this for use with OpenCode and cloud providers.

---

## Decisions

| Question | Decision |
|----------|----------|
| Provider set | Full: Ollama, OpenAI, local (sentence-transformers) for embeddings; Ollama, OpenAI, Anthropic for LLM |
| Interface contract | Duck typing with factory functions — no ABC/Protocol overhead |
| Package structure | `providers/` package with one file per provider |
| Dim mismatch handling | Detect at startup, refuse to start, explicit migration command |
| API key env vars | Standard (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) — no new MEMORY_V3_* key vars |

---

## Provider Package

```
src/memory_v3/providers/
  __init__.py        ← get_embedder() and get_llm() factories
  embedder_ollama.py
  embedder_openai.py
  embedder_local.py  ← sentence-transformers, no API key, lazy model load
  llm_ollama.py
  llm_anthropic.py   ← includes prompt caching for static prompt prefixes
  llm_openai.py
```

### Embedder interface (duck typed)

```python
class <Provider>Embedder:
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

### LLM interface (duck typed)

```python
class <Provider>LLM:
    def chat(self, prompt: str, max_tokens: int = 2048) -> str: ...
```

### Factory functions (`providers/__init__.py`)

```python
def get_embedder():
    provider = os.environ.get("MEMORY_V3_EMBED_PROVIDER", "ollama")
    if provider == "openai":   return OpenAIEmbedder()
    if provider == "local":    return LocalEmbedder()
    return OllamaEmbedder()

def get_llm():
    provider = os.environ.get("MEMORY_V3_LLM_PROVIDER", "ollama")
    if provider == "openai":     return OpenAILLM()
    if provider == "anthropic":  return AnthropicLLM()
    return OllamaLLM()
```

Unknown provider values raise `ValueError` with a message listing valid options.

---

## Config Changes

Two new fields in `config.py` `Config` dataclass:

```python
embed_provider: str = "ollama"   # from MEMORY_V3_EMBED_PROVIDER
llm_provider: str = "ollama"     # from MEMORY_V3_LLM_PROVIDER
```

User-facing config for switching to OpenAI + Anthropic (entire change):

```
MEMORY_V3_EMBED_PROVIDER=openai
MEMORY_V3_LLM_PROVIDER=anthropic
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Startup Dim Check

On server start (in `server.py` before registering MCP tools), check the `memory_vec` schema dim against `MEMORY_V3_EMBED_DIM`:

```python
def _check_embedding_dim(conn, configured_dim):
    # Query sqlite_master for the memory_vec CREATE statement
    # Parse float[N] to extract N
    # If N != configured_dim: print error and sys.exit(1)
```

Error message:

```
ERROR: Embedding dimension mismatch.
  DB was created with dim=768 (e.g. nomic-embed-text / Ollama).
  Current config requests dim=1536 (e.g. text-embedding-3-small / OpenAI).

This means all existing memories need to be re-embedded before the server can run.
Re-embedding sends every memory through your new embedding provider's API,
which may incur significant cost (e.g. ~$0.02 per 1M tokens for OpenAI —
costs add up quickly with thousands of memories).

To proceed intentionally, run:
  memory-v3-reindex --reembed

This will flush the embedding cache, rebuild memory_vec at the new dim,
and recompute all cluster centroids. Your memories are NOT deleted.
```

---

## Files Modified

| File | Change |
|------|--------|
| `embeddings.py` | Replace `ollama.embed()` with `get_embedder().embed()` / `embed_batch()`. Keep `embed_with_cache()` but invalidate cache on provider/dim change. |
| `lifecycle/extraction.py` | Replace `ollama.chat()` with `get_llm().chat()`. Remove top-level `import ollama`. |
| `lifecycle/compaction.py` | Replace `ollama.chat()` with `get_llm().chat()`. Remove top-level `import ollama`. |
| `config.py` | Add `embed_provider`, `llm_provider` fields. |
| `server.py` | Add `_check_embedding_dim()` call at startup. |
| `pyproject.toml` | Move `ollama>=0.4` from `dependencies` to `[project.optional-dependencies]`. Add `openai>=1.0`, `anthropic>=0.40` as optional extras. Add `sentence-transformers>=3.0` as optional. |

---

## Embedding Cache Invalidation

`embed_with_cache()` currently uses SHA-256 of text as cache key. This causes silent correctness bugs when switching providers — old embeddings are returned at the wrong dimension.

Fix: include provider name and dim in the cache key:

```python
cache_key = sha256(f"{provider}:{dim}:{text}".encode()).hexdigest()
```

This means cache entries from the old provider are simply not found (and regenerated), rather than returned incorrectly. Old cache files are orphaned but not harmful — they can be cleared with `memory-v3-reindex --reembed`.

---

## `memory-v3-reindex --reembed` Command

New flag on the existing `memory-v3-reindex` CLI command:

1. Print warning: lists memory count + estimated token count + reminder about API costs
2. Prompt for confirmation: `Re-embed N memories using <provider>? [y/N]`
3. Flush embedding cache directory
4. Drop and recreate `memory_vec` at the new dim
5. Re-embed all non-archived memories in batches of 50
6. Recompute all `cluster_centroids`
7. Print summary: memories re-embedded, time taken, errors if any

---

## pyproject.toml Extras

```toml
[project.optional-dependencies]
ollama = ["ollama>=0.4"]
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.40"]
local = ["sentence-transformers>=3.0"]
graph = ["python-igraph>=0.11", "leidenalg>=0.10", "pyvis>=0.3"]
dev = ["pytest>=8.0", "pytest-cov", "ruff"]
all = ["memory-v3[ollama,openai,anthropic,local,graph,dev]"]
```

Core `dependencies` list has `ollama` removed.

---

## Anthropic LLM Provider Notes

The Anthropic messages API differs from Ollama's shape:
- No `options` dict — use `temperature` and `max_tokens` as top-level params
- System prompt is a separate param, not a message role
- Response content is `.content[0].text`

The `AnthropicLLM.chat()` method will include **prompt caching** (`cache_control: {"type": "ephemeral"}`) on the static prompt prefix (the extraction/compaction instruction templates). This reduces cost significantly when `extract_from_conversation` is called repeatedly in a session.

---

## Local Embedder Configuration Note

The `local` provider uses sentence-transformers (no API key, runs on CPU). Different models output different dims, so users must also set `MEMORY_V3_EMBED_DIM` and `MEMORY_V3_EMBED_MODEL` to match. Recommended defaults:

| Model | Dim | Notes |
|-------|-----|-------|
| `all-mpnet-base-v2` | 768 | Same dim as Ollama nomic — no DB rebuild needed |
| `all-MiniLM-L6-v2` | 384 | Faster/smaller, requires DB rebuild |

Recommended default for `local` is `all-mpnet-base-v2` (768-dim) so users can switch from Ollama without triggering a reindex. The `LocalEmbedder` will print a warning if `MEMORY_V3_EMBED_DIM` doesn't match the loaded model's actual output dim.

---

## Out of Scope

- OpenCode-specific config documentation (already in CLAUDE.md)
- RL decider provider (experimental flag, unchanged)
- Streaming responses (none of the current prompts need it)
- Per-call provider override (always from env vars)
