# Memory v3 — Phase 03: Memory Lifecycle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic, atomic, filtered memory lifecycle behavior without touching the live owner database.

**Architecture:** Source files become immutable chunk sets attached to staged source generations. Activation switches visibility transactionally, while retrieval admits only policy-approved active rows before ranking. Direct user context remains an independently gated expiring handoff path.

**Tech Stack:** Python 3.14, SQLite/FTS5/sqlite-vec, FastMCP, pytest, NetworkX graph artifacts

---

## Planning Metadata

```yaml
spec: .planning/SPEC.md
phase: 03-memory-lifecycle
must_haves:
  truths:
    - "Final-quarter content from a long note is retrievable with line and heading provenance"
    - "Failed replacement preserves the prior active generation and file hash"
    - "Deleted sources leave no searchable derivatives or unreported graph debt"
    - "Access and source filters run before ranking and every result has provenance"
    - "Handoffs expire and durable writes require explicit administrative override"
    - "Interrupted indexing resumes without duplicate generations"
  artifacts:
    - path: "src/memory_v3/indexing.py"
      provides: "Chunking, generation staging, activation, checkpointing, and source cleanup"
    - path: "src/memory_v3/retrieval/filters.py"
      provides: "Mandatory retrieval eligibility filters"
    - path: "src/memory_v3/db.py"
      provides: "Schema v5, generation transactions, handoff and retrieval queries"
    - path: "src/memory_v3/server.py"
      provides: "Gated public APIs for indexing, retrieval, and handoffs"
  key_links:
    - from: "src/memory_v3/server.py"
      to: "src/memory_v3/indexing.py"
      via: "reindex stages and activates complete source generations"
    - from: "src/memory_v3/retrieval/stage1_coarse.py"
      to: "src/memory_v3/retrieval/filters.py"
      via: "candidate eligibility is intersected before ranking"
    - from: "src/memory_v3/indexing.py"
      to: "src/memory_v3/build_kg.py"
      via: "source deletion reports or runs deterministic active-generation graph rebuild"
hard_boundaries:
  - "Never open the live owner DB or graph paths in migration, tests, or smoke checks"
  - "Never expose staged or superseded generations to normal retrieval"
  - "Never advance file hashes or checkpoints after provider failure"
  - "Each task changes at most five files and passes independent review"
waves:
  wave_1: ["Task 1", "Task 2"]
  wave_2: ["Task 3", "Task 4", "Task 5"]
  wave_3: ["Task 6", "Task 7A", "Task 7B"]
  wave_4: ["Task 8A", "Task 8B", "Task 9", "Task 10"]
  wave_5: ["Task 11"]
```

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/memory_v3/db.py` | Modify | Schema v5, atomic generation, cleanup, retrieval, handoff queries |
| `src/memory_v3/migration.py` | Modify | Explicit additive v4-to-v5 migration |
| `src/memory_v3/indexing.py` | Create | Chunking, staging, activation, resume, deletion orchestration |
| `src/memory_v3/server.py` | Modify | Public indexing, retrieval, and handoff APIs |
| `src/memory_v3/source_policy.py` | Modify | Historical/ranking metadata used by indexing and retrieval |
| `src/memory_v3/retrieval/filters.py` | Create | Mandatory SQL/candidate eligibility filters |
| `src/memory_v3/retrieval/stage1_coarse.py` | Modify | Filter candidates before scoring |
| `src/memory_v3/retrieval/stage2_fine.py` | Modify | Rank only eligible rows and apply rank weights |
| `src/memory_v3/retrieval/stage3_organize.py` | Modify | Prevent graph/link annotations from reintroducing excluded rows |
| `src/memory_v3/multi_agent.py` | Modify | Active-handoff synchronization and legacy fallback |
| `src/memory_v3/embeddings.py` | Modify | Bounded retry integration |
| `src/memory_v3/providers/embedder_openai.py` | Modify | Retryable provider error classification |
| `src/memory_v3/config.py` | Modify | Retry and legacy-sync settings |
| `src/memory_v3/cli.py` | Modify | Tested lifecycle and handoff commands |
| `AGENTS.md` | Modify | Updated startup and compatibility guidance |

---

## Task 1: Add schema v5 generation foundation

**Files:** `src/memory_v3/db.py`, `src/memory_v3/migration.py`, `tests/test_db.py`, `tests/test_migration.py`

- [x] Add failing fresh-schema tests for generation, chunk-location, rank, historical, and deletion fields.
- [x] Add failing synthetic v4-to-v5 migration, idempotence, rollback, and data-preservation tests.
- [x] Add `source_generations`, `staged_source_chunks`, `indexing_checkpoints`, and `indexing_debt` tables plus generation-aware `file_hashes` fields.
- [x] Store staged content, metadata, and serialized embedding only in `staged_source_chunks`; staged rows must never enter active `memories`, `memory_vec`, or FTS tables.
- [x] Add only additive columns and non-destructive indexes; do not rewrite vectors, FTS, IDs, or content.
- [x] Keep normal startup migration-free and require explicit migration.
- [x] Run focused DB/migration tests, Ruff, compilation, and independent review.

## Task 2: Implement deterministic Markdown chunking

**Files:** `src/memory_v3/indexing.py`, `src/memory_v3/source_policy.py`, `tests/test_indexing_chunks.py`, `tests/test_reindex_policy.py`

- [x] Write failing tests for multiple chunks, final-quarter coverage, heading stacks, exact line spans, and bounded overlap.
- [x] Freeze configurable defaults at `target_chars=3500`, `max_chars=5000`, and `overlap_lines=3` unless tests expose a safer boundary.
- [x] Use exact identities: `source_id = sha256(owner + namespace + canonical_path)`, `generation_key = sha256(source_id + source_hash)`, and `chunk_id = sha256(generation_key + chunk_index + line_start + line_end + chunk_content_hash)`.
- [x] Prove changed content creates a new generation and non-conflicting chunk IDs while unchanged content reuses identical IDs.
- [x] Keep the chunker pure: no file IO, DB, providers, or global config reads.
- [x] Prove two unchanged runs produce byte-identical chunk metadata.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 3: Stage and atomically activate source generations

**Files:** `src/memory_v3/indexing.py`, `src/memory_v3/server.py`, `src/memory_v3/db.py`, `tests/test_reindex_generations.py`, `tests/test_reindex_policy.py`

- [x] Write failing tests for staged invisibility, atomic activation, failed replacement preservation, and source-specific errors.
- [x] Stage every embedded chunk in `staged_source_chunks`, never active memory/vector/FTS tables, before changing visibility or `file_hashes`.
- [x] Activate with one `BEGIN IMMEDIATE` transaction and preserve the prior active generation on any failure.
- [x] During activation, copy the complete staged set into active memory/vector/FTS storage, switch the generation, update `file_hashes`, and remove staging rows in the same transaction.
- [x] Prove raw keyword, vector, hybrid, CLI, and MCP search cannot see staged chunks.
- [x] Enforce `GENERATION_ACTIVATION_ENABLED` independently from indexing.
- [x] Persist checkpoints after complete staged chunks and resume without duplicate generations or chunks.
- [x] Prove forced unchanged reindex has stable identities and counts, while old active and new staged generations coexist without key collisions.
- [x] Return source errors with `source_path`, `source_id`, `stage`, `retryable`, and explicit active-generation/file-hash unchanged flags.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 4: Delete sources and clean every derivative

**Files:** `src/memory_v3/indexing.py`, `src/memory_v3/db.py`, `src/memory_v3/retrieval/stage3_organize.py`, `tests/test_source_deletion_cleanup.py`, `tests/test_reindex_generations.py`

- [x] Write failing tests for memory, vector, FTS, file hash, checkpoint, generation, link, community, centroid, router-cache, and graph contribution cleanup/invalidation.
- [x] Tombstone generation metadata while deleting searchable row/vector/FTS derivatives transactionally.
- [x] Scrub deleted memory IDs from surviving zettelkasten links.
- [x] Insert measurable `indexing_debt(kind='graph_rebuild_required', source_id, created_at)` when safe source-aware pruning is impossible.
- [x] Make stage-3 graph/community/link enrichment skip all graph-derived additions while unresolved rebuild debt exists.
- [x] Leave measurable rebuild debt for Task 5; do not claim graph/community/cache cleanup inside the source-row transaction.
- [x] Stop if any deleted source remains searchable, graph/community/cache debt is absent, or stale enrichment remains usable.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 5: Fail closed on graph/community/cache debt

**Files:** `src/memory_v3/indexing.py`, `src/memory_v3/db.py`, `src/memory_v3/build_kg.py`, `src/memory_v3/graphs/communities.py`, `tests/test_graph_debt.py`

- [x] Write failing tests proving graph, community, centroid, and routing/cache artifacts are rebuilt without deleted source contributions.
- [x] Rebuild graph, community, centroid, and routing/cache artifacts only from active generations when graph mutation is enabled.
- [x] Clear rebuild debt only after a complete successful rebuild; preserve debt on failure or when graph mutation is disabled.
- [x] Assert exact artifact paths/state before and after rebuild in temporary graph directories.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 6: Apply mandatory retrieval filters before ranking

**Files:** `src/memory_v3/retrieval/filters.py`, `src/memory_v3/retrieval/stage1_coarse.py`, `src/memory_v3/retrieval/stage2_fine.py`, `src/memory_v3/db.py`, `tests/test_retrieval_filters.py`

- [x] Write failing vector-only, FTS-only, mixed, and cross-owner isolation tests.
- [x] Implement typed filters for owner, namespace, project, path prefix, author, sensitivity, record kind, active generation, and historical mode.
- [x] Apply eligibility inside raw FTS SQL before rank/limit; for sqlite-vec, over-fetch bounded candidates and intersect with eligible IDs before any application scoring or top-N exposure, with tests proving ineligible candidates never reach rankers.
- [x] Apply the same eligibility to `hybrid_search`, `keyword_search`, and staged retrieval.
- [x] Return source hash/path, owner, namespace, dates, verification, chunk lines/headings, generation, and record kind.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 7A: Index historical research and apply post-filter down-ranking

**Files:** `src/memory_v3/config.py`, `src/memory_v3/source_policy.py`, `src/memory_v3/indexing.py`, `src/memory_v3/retrieval/stage2_fine.py`, `tests/test_retrieval_historical.py`

- [x] Add fail-closed `MEMORY_V3_RETRIEVAL_V2_ENABLED` configuration for Task 7B public enforcement.
- [x] Write failing tests for paused versus active ranking, terminal exclusion, historical opt-in, and current research older than 180 days versus recent research.
- [x] Index archived/superseded research as `historical=1` in the same source-generation pipeline so explicit historical retrieval has retained data; normal filters must still exclude it.
- [x] Keep paused projects eligible with `rank_weight < 1.0` after mandatory filters.
- [x] Apply a documented `rank_weight < 1.0` to non-historical research older than 180 days after mandatory filters.
- [x] Exclude archived/superseded research by default and include it only with explicit `historical=true`.
- [x] Preserve complete provenance in all historical and normal results.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 7B: Enforce public retrieval rollback and graph-neighbor eligibility

**Files:** `src/memory_v3/server.py`, `src/memory_v3/retrieval/__init__.py`, `src/memory_v3/retrieval/stage3_organize.py`, `tests/test_retrieval_public.py`

- [x] Fail closed under `MEMORY_V3_RETRIEVAL_V2_ENABLED=false` before DB, embedding, graph, or provider access without changing indexed data.
- [x] Make configured owner mandatory on every public search path; reject or zero-result cross-owner requests.
- [x] Expose namespace, project, path, author, sensitivity, record-kind, and explicit historical filters through public APIs.
- [x] Pass one immutable filter through Stage 1, Stage 2, and Stage 3.
- [x] Filter graph and zettelkasten neighbors before enrichment so related records cannot reintroduce excluded owners, paths, statuses, or historical rows.
- [x] Prove public keyword, hybrid, staged, CLI-compatible, and MCP search results all contain complete provenance.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 8A: Complete handoff lifecycle and active synchronization

**Files:** `src/memory_v3/config.py`, `src/memory_v3/server.py`, `src/memory_v3/db.py`, `src/memory_v3/multi_agent.py`, `tests/test_handoffs.py`

- [x] Write failing tests for 30-day defaults, durable override rejection/acceptance, expiry, and disabled sweeping.
- [x] Require explicit `durable=true` plus administrative override for durable direct writes.
- [x] Sweep only expired `handoff` rows when `HANDOFF_SWEEPER_ENABLED=true`; never delete durable or derived rows.
- [x] Add active-handoff retrieval with optional agent/since filtering.
- [x] Make `agent_sync` return active handoffs with deprecation metadata; legacy changelog behavior requires an explicit compatibility setting.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 8B: Update startup guidance

**Files:** `AGENTS.md`

- [x] Remove mandatory generic search, recent listing, changelog sync, and blanket conversation extraction from startup/session-end guidance.
- [x] Document active handoff synchronization and the explicit disabled-by-default legacy fallback.
- [x] Confirm commands and environment names match tested Task 8A behavior.

## Task 9: Fix UTC recent queries and legacy compatibility

**Files:** `src/memory_v3/db.py`, `src/memory_v3/server.py`, `src/memory_v3/multi_agent.py`, `tests/test_recent_and_sync.py`

- [x] Write a two-hour-old fixture and prove `list_recent(hours=1)` excludes it.
- [x] Use timezone-aware UTC cutoffs and parameterized SQL.
- [x] Prove recent active handoffs remain visible and expired handoffs do not.
- [x] Document and test the explicit legacy `agent_sync` fallback.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 10: Add bounded embedding retry and resume safety

**Files:** `src/memory_v3/embeddings.py`, `src/memory_v3/providers/embedder_openai.py`, `src/memory_v3/config.py`, `tests/test_embedding_retry.py`

- [x] Write failing 429, 5xx, non-retryable, exhaustion, and bounded-sleep tests.
- [x] Add configurable attempts and capped exponential backoff with injectable sleep.
- [x] Enforce a configurable maximum total retry sleep so rate-limit handling is bounded in both attempts and elapsed delay.
- [x] Retry only recognized transient failures and fail immediately otherwise.
- [x] Prove retry exhaustion leaves no partial active generation or advanced file hash.
- [x] Run focused tests, Ruff, compilation, and independent review.

## Task 11: Integrate CLI, kill-switch, rollback, and end-to-end gates

**Files:** `src/memory_v3/cli.py`, `src/memory_v3/server.py`, `tests/test_reindex_policy.py`, `tests/test_handoffs.py`, `AGENTS.md`

- [x] Add tested commands for forced reindex, handoff listing/sweeping, and optional graph rebuild.
- [x] Prove every kill switch blocks before reads, DB, providers, graph initialization, or mutation.
- [x] Run unchanged reindex twice and compare generations, chunks, identities, and counts.
- [x] Run fixed canaries for final-quarter retrieval, failed replacement, deletion cleanup, cross-owner isolation, historical mode, expiry, recent cutoff, and resume.
- [x] Prove Phase 3A rollback leaves active generation ID, file hash, and active row/vector/FTS counts unchanged when activation is disabled or replacement fails.
- [x] Prove Phase 3B retrieval rollback disables new endpoints without changing indexed data.
- [x] Prove Phase 3C direct-write and sweeper switches independently produce zero writes/deletions when disabled.
- [x] Prove Phase 3D retry exhaustion leaves no partial active generation, advanced file hash, or changed active counts.
- [x] Run the full repository test suite, touched-file Ruff/format/compile checks, and `git diff --check` using only `/tmp/opencode` DB/vault/graph paths.
- [x] Run final adversarial review and record evidence in the vault before any Phase 4 shadow build.
