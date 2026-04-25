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
  embeddings.py      # Embedding layer — delegates to providers/get_embedder()
  security.py        # Credential scanning (regex)
  cli.py             # CLI commands
  build_kg.py        # Graph construction from existing memories
  migration.py       # v2 -> v3 migration

  providers/         # Provider abstraction (ollama/openai/anthropic/local)
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
    extraction.py    # 2-pass LLM extraction — delegates to providers/get_llm()
    compaction.py    # CogCanvas compression — delegates to providers/get_llm()
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

## Provider Abstraction (Completed)

Ollama dependency removed from core. All embedding and LLM calls go through `providers/get_embedder()` and `providers/get_llm()`. Select provider via env vars:

```
MEMORY_V3_EMBED_PROVIDER=ollama|openai|local   # default: ollama
MEMORY_V3_LLM_PROVIDER=ollama|openai|anthropic # default: ollama
```

### Dimension mismatch

`memory_vec` dim is fixed at DB creation. Switching providers with different dims requires `memory-v3 reindex --reembed` (drops and rebuilds the vector table — prompts for confirmation). Server startup will `sys.exit(1)` if configured dim doesn't match the DB.

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

OpenCode v1.14+ format — key differences from older docs: top-level key is `mcp` (not `mcpServers`), `type` is `"local"`, `command` is an array, env key is `environment`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "anthropic/claude-sonnet-4-5",
  "mcp": {
    "memory-v3": {
      "type": "local",
      "command": ["/path/to/venv/bin/memory-v3-server"],
      "enabled": true,
      "environment": {
        "MEMORY_V3_DB": "/path/to/memory.db",
        "MEMORY_V3_VAULT": "/path/to/vault",
        "MEMORY_V3_EMBED_PROVIDER": "openai",
        "MEMORY_V3_LLM_PROVIDER": "openai",
        "MEMORY_V3_EMBED_MODEL": "text-embedding-3-small",
        "MEMORY_V3_EMBED_DIM": "1536",
        "MEMORY_V3_LLM_MODEL": "gpt-4o-mini"
      }
    }
  }
}
```

## Workflow Preferences

- When executing implementation plans, always use **subagent-driven development** (option 1) — fresh subagent per task with spec + code quality review after each.

## Gotchas

- `config.py` uses a cached singleton via `get_config()` — call `reset_config()` in tests after setting env vars.
- Provider imports are lazy (inside each provider class) — missing optional packages raise a clear ImportError at first use, not at server start.
- `memory_vec` dim is fixed at DB creation time — cannot be altered in-place, use `memory-v3 reindex --reembed`.
- Graph state is serialized to `~/.memory-v3/graphs/` — if you change graph schema, delete these files and let them rebuild on next run.
- `ollama` is now an optional extra (`pip install "memory-v3[ollama]"`), not a core dependency.
