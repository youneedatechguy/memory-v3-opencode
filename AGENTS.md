# memory-v3-opencode — OpenCode Session Protocol

## Session Startup

At the start of each session:
1. Call `agent_sync("opencode")` to catch up on memory changes from other agents
2. Call `search("current project context")` to recall relevant memories
3. Call `list_recent` (last 24h) if returning to in-progress work

## Session End

When switching tasks or ending a session:
1. Call `extract_from_conversation` with a summary of what was decided/built

## Provider Config Reference

- Embeddings: OpenAI `text-embedding-3-small`, dim=1536
- LLM: Anthropic `claude-haiku-4-5-20251001` (fast, cheap, handles extraction/compaction)
- Agent LLM: Anthropic Claude Sonnet (full reasoning, tool orchestration — handled by OpenCode itself)

---

# memory-v3-opencode

Fork of [Haustorium12/memory-v3](https://github.com/Haustorium12/memory-v3). Goal: replace the hard Ollama dependency with a pluggable provider system supporting OpenAI-compatible embeddings, Anthropic/OpenAI LLMs, and local sentence-transformers. Target runtime: [OpenCode](https://opencode.ai) (MCP stdio/SSE).

## Commands

```bash
# Install (editable, all extras)
pip install -e ".[all]"

# Start MCP server
memory-v3-server

# CLI
memory-v3 stats
memory-v3 search "query"
memory-v3 add "fact" --type fact --tags tag1,tag2
memory-v3 recent --hours 48

# Build knowledge graphs from existing memories
memory-v3-build-kg

# Migrate from v2
memory-v3-migrate

# Lint
ruff check src/
ruff format src/

# Tests
pytest
pytest tests/test_scoring/  # scoring subsystem only
```

## Architecture

```
src/memory_v3/
  server.py          # FastMCP server — 24 MCP tool endpoints (entry point)
  db.py              # SQLite schema, CRUD, FTS5 + sqlite-vec search
  config.py          # All config via MEMORY_V3_* env vars, get_config() singleton
  embeddings.py      # Embedding layer — currently Ollama-only (TARGET FOR REFACTOR)
  security.py        # Credential scanning (regex)
  cli.py             # CLI commands
  build_kg.py        # Graph construction from existing memories
  migration.py       # v2 -> v3 migration

  providers/         # *** TO BE CREATED — provider abstraction ***
    __init__.py      # factory: get_embedder(), get_llm()
    embedder_ollama.py
    embedder_openai.py
    embedder_local.py   # sentence-transformers
    llm_ollama.py
    llm_anthropic.py
    llm_openai.py

  async_ops/
    write_queue.py   # Priority queue (1-5), batch flush, retry
    batch_embedder.py

  graphs/            # MAGMA — 4 NetworkX DiGraph layers
    __init__.py      # GraphManager
    semantic.py      # related_to, uses, depends_on, etc.
    temporal.py      # preceded_by, followed_by, etc.
    causal.py        # caused, enabled, prevented, etc.
    entity.py        # built, maintains, responsible_for, etc.
    zettelkasten.py  # Auto bidirectional linking (cosine > 0.5)
    communities.py   # Leiden community detection + centroid routing
    router.py        # Query -> graph layer routing

  lifecycle/
    extraction.py    # 2-pass LLM extraction — calls ollama.chat() (TARGET FOR REFACTOR)
    compaction.py    # CogCanvas compression — calls ollama.chat() (TARGET FOR REFACTOR)
    consolidation.py # 4-phase sleep cycle
    sensory.py       # 6-gate intake filter
    hallucination.py # HaluMem confidence decay + verification
    action_decider.py

  retrieval/
    stage1_coarse.py # Community routing + vec0 + FTS5 (up to 500 candidates)
    stage2_fine.py   # RRF + ACT-R + fan effect + HaluMem confidence scoring
    stage3_organize.py # Graph neighbors + structure hints + zettelkasten see-also

  scoring/
    actr.py          # ACT-R base-level + spreading activation + noise
    fademem.py       # Importance scoring, Weibull decay, archival logic
    surprise.py      # Centroid-based surprise (Titans-inspired)
    hierarchy.py     # Governance layer auto-classification
    fan_effect.py    # ACT-R IDF-equivalent fan penalty
```

## Current Refactor: Provider Abstraction

**Goal:** decouple Ollama from the core so any embedding/LLM provider works.

### Files to modify

| File | What changes |
|------|-------------|
| `embeddings.py` | Replace `ollama.embed()` calls with `get_embedder().embed()` |
| `lifecycle/extraction.py` | Replace `ollama.chat()` with `get_llm().chat()` |
| `lifecycle/compaction.py` | Replace `ollama.chat()` with `get_llm().chat()` |
| `config.py` | Add `MEMORY_V3_EMBED_PROVIDER` and `MEMORY_V3_LLM_PROVIDER` env vars |
| `pyproject.toml` | Make `ollama` optional; add `openai`, `anthropic` as optional extras |

### New env vars (to add)

```
MEMORY_V3_EMBED_PROVIDER=ollama|openai|local   # default: ollama
MEMORY_V3_LLM_PROVIDER=ollama|openai|anthropic # default: ollama
```

### Dimension mismatch warning

`memory_vec` is created with `float[768]` fixed at DB creation. Switching providers that use different dims (e.g. OpenAI `text-embedding-3-small` = 1536) requires dropping and rebuilding `memory_vec`. A `memory-v3-reindex --reembed` migration command is planned.

## Key Config Env Vars

```
MEMORY_V3_DB          Path to SQLite DB (default: ~/.memory-v3/memory.db)
MEMORY_V3_VAULT       Markdown vault directory
MEMORY_V3_EMBED_MODEL Embedding model name (default: nomic-embed-text)
MEMORY_V3_LLM_MODEL   LLM model name (default: qwen2.5:3b)
MEMORY_V3_EMBED_DIM   Embedding dimensions (default: 768)
```

Feature flags all follow `MEMORY_V3_ENABLE_<NAME>=true|false`.

## OpenCode MCP Config

```json
{
  "mcpServers": {
    "memory-v3": {
      "type": "stdio",
      "command": "memory-v3-server",
      "env": {
        "MEMORY_V3_DB": "/path/to/memory.db",
        "MEMORY_V3_EMBED_PROVIDER": "openai",
        "MEMORY_V3_LLM_PROVIDER": "anthropic"
      }
    }
  }
}
```

## Workflow Preferences

- When executing implementation plans, always use **subagent-driven development** (option 1) — fresh subagent per task with spec + code quality review after each.

## Gotchas

- `config.py` uses a cached singleton via `get_config()` — call `reset_config()` in tests after setting env vars.
- `ollama` is imported at module top-level in `embeddings.py`, `extraction.py`, and `compaction.py` — provider swap must move these imports inside provider classes or use lazy imports to avoid ImportError when Ollama isn't installed.
- `memory_vec` dim is fixed at DB creation time — cannot be altered in-place, must drop and rebuild.
- Graph state is serialized to `~/.memory-v3/graphs/` — if you change graph schema, delete these files and let them rebuild on next run.
- `pyproject.toml` pins `ollama>=0.4` in core `dependencies` — must move to optional after provider abstraction.
