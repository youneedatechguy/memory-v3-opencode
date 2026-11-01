# memory-v3-opencode — OpenCode Session Protocol

## Session Startup

At the start of each session:
1. Call `list_active_handoffs(agent_id="opencode")` for temporary cross-session state from other active agents.
2. Search memory only for a specific fact or project question; do not run generic startup searches.
3. Do not call `list_recent` as a default startup step.

`agent_sync` is a deprecated compatibility API. Its default mode returns active handoffs only; legacy changelog sync is removed.

## Session End

When switching tasks or ending a session:
1. Put durable information in canonical vault/project docs first.
2. Use a direct handoff only for temporary cross-session state that another agent needs; default expiry is 30 days (`MEMORY_V3_HANDOFF_TTL_DAYS`).
3. Do not call `extract_from_conversation` as a blanket session-end requirement.

Direct writes, vault auto-extract, and the handoff sweeper are independently kill-switched by `MEMORY_V3_DIRECT_WRITES_ENABLED`, `MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED`, and `MEMORY_V3_HANDOFF_SWEEPER_ENABLED`.

## Provider Config Reference

- Embeddings: OpenAI `text-embedding-3-small`, dim=1536
- LLM: OpenAI `gpt-4.1-nano` (fast, cheap, handles extraction/compaction)
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
memory-v3 reindex --force
memory-v3 handoffs list --agent-id opencode --since 2026-10-02T00:00:00+00:00
memory-v3 handoffs sweep

# Deprecated graph command retained for parser compatibility; always exits nonzero
memory-v3 build-kg

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
  server.py          # FastMCP server — 26 MCP tool endpoints (entry point)
  db.py              # SQLite schema v4 fresh-shadow target, CRUD, FTS5 + sqlite-vec search
  config.py          # All config via MEMORY_V3_* env vars, get_config() singleton
  embeddings.py      # Embedding layer — delegates to providers/get_embedder()
  security.py        # Credential scanning (regex)
  cli.py             # CLI commands
  build_kg.py        # Dormant legacy graph code; not part of active retrieval/indexing
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

  graphs/            # Dormant legacy graph modules; graph artifacts are inactive
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

  indexing.py        # Chunk files and atomically replace each source in one transaction

  retrieval/
    filters.py       # Mandatory owner/source eligibility before ranking
    stage1_coarse.py # Lexical/vector candidates, graph-free
    stage2_fine.py   # RRF + ACT-R + fan effect + HaluMem confidence scoring
    stage3_organize.py # Graph-free result organization and provenance

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

`memory_vec` dim is fixed at DB creation. `memory-v3 reindex --reembed` is retained only for parser compatibility and fails closed before provider imports or DB access. Switching providers with different dims requires the explicit fresh-shadow migration/reindex workflow. Server startup will `sys.exit(1)` if configured dim doesn't match the DB.

## Lean v4 Indexing and Retrieval

- Schema v4 is the fresh-shadow target. Normal startup rejects incompatible schemas rather than performing an in-place downgrade.
- Vault indexing builds all chunks and embeddings for a source first, then atomically replaces that source's memories, vectors, FTS rows, and file hash rows in one SQLite transaction.
- Retrieval is graph-free and uses mandatory owner/source eligibility before lexical/vector ranking.
- Direct writes support temporary handoffs (30-day TTL) and durable facts (no expiry, namespace=durable) via `durable=True`, plus admin overrides via `admin_override=True`.
- Graph modules and graph artifacts are dormant. `build-kg` is hard-disabled and `graph-debt` is not a CLI command.

## Key Config Env Vars

```
MEMORY_V3_DB          Path to SQLite DB (default: ~/.memory-v3/memory.db)
MEMORY_V3_VAULT       Markdown vault directory
MEMORY_V3_EMBED_MODEL Embedding model name (default: nomic-embed-text)
MEMORY_V3_LLM_MODEL   LLM model name (default: gpt-4.1-nano)
MEMORY_V3_EMBED_DIM   Embedding dimensions (default: 768)
MEMORY_V3_INDEXING_ENABLED              Enable vault indexing/reindex
MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED   Enable policy-controlled vault auto-extract
MEMORY_V3_DIRECT_WRITES_ENABLED         Enable direct write endpoints
MEMORY_V3_HANDOFF_SWEEPER_ENABLED       Enable deletion of expired handoffs
MEMORY_V3_HANDOFF_TTL_DAYS              Handoff expiry window in days
MEMORY_V3_SOURCE_POLICY_OWNER           Owner assigned to indexed vault sources
MEMORY_V3_SOURCE_POLICY_INCLUDE_ROOTS   Comma-separated vault roots to index
MEMORY_V3_SOURCE_POLICY_DENY_ROOTS      Comma-separated vault roots to exclude
```

Legacy generation, retrieval-v2, graph mutation/debt, durable direct-write override, historical retention/rank, and legacy sync flags are not supported.

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
        "MEMORY_V3_LLM_MODEL": "gpt-4.1-nano"
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
- `memory_vec` dim is fixed at DB creation time — cannot be altered in-place; use the explicit fresh-shadow migration/reindex workflow.
- Graph state under `~/.memory-v3/graphs/` is dormant and must not be used by active indexing or retrieval.
- `ollama` is now an optional extra (`pip install "memory-v3[ollama]"`), not a core dependency.
