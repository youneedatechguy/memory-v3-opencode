# Memory v3 — Phase 3 Specification

## Source

Approved Phase 3 requirements are maintained in `/home/base/vault/projects/agent-architecture/plans/CURRENT.md` under “Phase 3 — Memory indexing, lifecycle, retrieval, and handoffs”.

## Goal

Turn the Phase 2 policy/schema foundation into a reliable source lifecycle with deterministic chunking, atomic generations, mandatory pre-ranking retrieval filters, expiring handoffs, and resumable provider-safe indexing.

## Required outcomes

- Long Markdown sources are indexed as deterministic heading-aware, line-addressable chunks with bounded overlap.
- A complete replacement is staged before one transaction activates it.
- Failed or interrupted replacement leaves the prior active generation and source hash unchanged.
- Changed and deleted sources leave no searchable stale memory, vector, FTS, file-hash, link, graph, community, or cache derivatives.
- Unsafe graph pruning is replaced by deterministic rebuild from active generations.
- Retrieval filters owner, namespace, project, path, author, sensitivity, record kind, generation status, and historical mode before ranking.
- A retrieval-specific rollback switch disables new retrieval behavior without modifying indexed data.
- Every result returns source and chunk provenance.
- Paused projects and current research older than 180 days remain searchable but down-ranked; historical research requires explicit opt-in.
- Direct writes default to 30-day handoffs; durable writes require explicit administrative override.
- Expired handoffs are swept only when the sweeper switch is enabled.
- Active handoffs replace changelog-based startup synchronization; legacy `agent_sync` remains a documented deprecated fallback.
- `list_recent(hours=...)` applies a real UTC predicate.
- Embedding retries are bounded for retryable 429/5xx failures.
- Interrupted indexing resumes without duplicate generations.

## Safety boundaries

- Never migrate, index, or test against the live owner database or graph paths.
- Existing v4 databases are migrated only through explicit migration functions against synthetic or copied databases.
- Staged chunks live outside active memory, vector, and FTS tables until one activation transaction copies them into active storage.
- No network provider calls in tests.
- Every implementation task touches at most five files and receives an independent review.
- Phase 2 kill switches remain fail closed and independently enforceable.
- Stop if deletion is incomplete, activation is non-atomic, filtering occurs after ranking, or a provider failure changes the active generation.
