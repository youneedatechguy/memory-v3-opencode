# OpenCode Integration Design

**Date:** 2026-04-25
**Repo:** memory-v3-opencode
**Goal:** Run memory-v3 as the persistent memory backend for OpenCode instead of Claude Code.

---

## What Changes

The memory-v3 MCP server itself is unchanged — it speaks MCP over stdio/SSE regardless of host. What changes is:

1. **Config file location and format** — OpenCode uses its own JSON config, not `.claude/settings.json`
2. **Tool timeout risk** — two lifecycle tools run synchronously and can be slow
3. **Project instructions** — OpenCode has its own equivalent of `CLAUDE.md`
4. **Provider wiring** — OpenCode handles the agent LLM; memory-v3 handles its own internal LLM separately

---

## OpenCode Installation (Linux)

```bash
# Via npm (recommended)
npm install -g opencode-ai

# Or via the install script
curl -fsSL https://opencode.ai/install | bash
```

Verify:
```bash
opencode --version
```

---

## Config File

OpenCode config lives at `~/.config/opencode/config.json`.

Full working config for this project (OpenCode v1.14+):

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
        "MEMORY_V3_DB": "/home/base/.memory-v3/memory.db",
        "MEMORY_V3_VAULT": "/home/base/vault",
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

Find your server binary path with:
```bash
which memory-v3-server
# or if using a venv:
/path/to/venv/bin/python -c "import shutil; print(shutil.which('memory-v3-server'))"
```

**Key format differences from older OpenCode versions:**
- Top-level key is `mcp`, not `mcpServers`
- `type` must be `"local"` (not `"stdio"`)
- `command` must be an **array** (not a string)
- Environment key is `environment` (not `env`)

**Provider rationale:**
- OpenCode agent → Anthropic Claude Sonnet (full reasoning, tool orchestration)
- memory-v3 internal LLM → OpenAI gpt-4o-mini (fast, cheap, handles extraction/compaction prompts)
- memory-v3 embeddings → OpenAI text-embedding-3-small (1536-dim, strong quality, no local GPU needed)

**Note:** Set `OPENAI_API_KEY` in `~/.bashrc` or `~/.profile` — OpenCode inherits env vars from the shell that launched it. Do not hardcode keys in the config file.

---

## Tool Timeout Risk

Two memory-v3 lifecycle tools run LLM calls synchronously inside the MCP tool response:

| Tool | Why it's slow | Typical duration |
|------|--------------|-----------------|
| `extract_from_conversation` | 2-pass LLM extraction (extract facts, then decide action for each) | 5–30s depending on text length |
| `compact_text` | LLM summarisation + faithfulness verification | 5–20s |

If OpenCode enforces a short MCP tool timeout these will fail.

**Mitigation (to implement):** Add a `--async` flag to both tools that queues the operation in the write queue and returns immediately with a job ID. The agent can call `stats` to check queue depth. This is a code change in `server.py` and `lifecycle/extraction.py` / `lifecycle/compaction.py`.

**Workaround until then:** Call these tools early in a session before doing other work, so they can run without blocking.

---

## Project Instructions (CLAUDE.md Equivalent)

OpenCode reads a file called `AGENTS.md` from the project root as its equivalent of `CLAUDE.md`. It is auto-loaded into the system prompt at session start.

Action: rename / copy `CLAUDE.md` to `AGENTS.md` in the repo root. Keep `CLAUDE.md` for Claude Code compatibility.

```bash
cp CLAUDE.md AGENTS.md
```

Add an OpenCode-specific section to `AGENTS.md`:

```markdown
## OpenCode Session Startup

At the start of each session:
1. Call `agent_sync("opencode")` to catch up on memory changes from other agents
2. Call `search("current project context")` to recall relevant memories
3. Call `list_recent` (last 24h) if returning to in-progress work

At session end (or when switching tasks):
1. Call `extract_from_conversation` with a summary of what was decided/built
```

---

## Memory DB Location

Use a shared DB path so memories persist across sessions and are accessible from both OpenCode and Claude Code during transition:

```
~/.memory-v3/memory.db
```

This is already the default. No change needed unless running multiple isolated projects.

---

## Verifying the Integration

After installing and configuring:

```bash
# Start OpenCode in the project directory
cd /home/base/memory-v3-opencode
opencode

# In the OpenCode session, test MCP tools are available:
# Ask: "What memory tools do you have access to?"
# Expected: lists all 24 memory-v3 tools

# Test a round-trip:
# Ask: "Add a memory: this project uses OpenAI embeddings at 1536 dim"
# Then: "Search for embedding configuration"
# Expected: returns the memory you just added
```

---

## Migration from Claude Code

| Item | Claude Code | OpenCode |
|------|-------------|----------|
| MCP config | `.claude/settings.json` → `mcpServers` | `~/.config/opencode/config.json` → `mcp` (type: local, command: array) |
| Project instructions | `CLAUDE.md` | `AGENTS.md` (copy from CLAUDE.md) |
| Memory DB | `~/.memory-v3/memory.db` | Same path — shared |
| Agent sync ID | `"claude-code"` | `"opencode"` (pass to `agent_sync`) |
| Permissions model | Per-tool allow/deny in settings | OpenCode auto-approves MCP tools |

Both tools can run against the same memory DB simultaneously — `agent_sync` and the multi-agent changelog handle conflict resolution.

---

## Out of Scope

- OpenCode keybindings and UI customisation
- Running memory-v3 as an SSE server (stdio is sufficient for single-user)
- Windows/Mac installation (user is on Linux)
- Migrating existing Claude Code conversation history into memories (separate task)
