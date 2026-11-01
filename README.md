<p align="center">
  <h1 align="center">memory-v3</h1>
  <p align="center">
    Brain-inspired persistent memory for AI coding assistants
  </p>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pypi.org/project/memory-v3/"><img src="https://img.shields.io/badge/pypi-v1.0.0-orange.svg" alt="PyPI"></a>
  <a href="#mcp-server-and-tools"><img src="https://img.shields.io/badge/MCP%20tools-24-green.svg" alt="24 MCP Tools"></a>
  <a href="#multi-graph-architecture"><img src="https://img.shields.io/badge/graphs-4%20layers-purple.svg" alt="4 Graph Layers"></a>
</p>

---

## Why memory-v3?

| Feature | mem0 | Zep | LangMem | **memory-v2** | **memory-v3** |
|---|---|---|---|---|---|
| Persistent local storage | -- | Cloud | Cloud | SQLite | SQLite + WAL |
| Vector search | Qdrant/Pinecone | Postgres | Pinecone | sqlite-vec | sqlite-vec + FTS5 + community routing |
| Knowledge graph | -- | -- | -- | Single NetworkX | **4-layer MAGMA** (semantic/temporal/causal/entity) |
| Retrieval pipeline | Vector only | Vector + keyword | Vector | Hybrid (BM25 + vec) | **3-stage** (coarse -> fine -> organize) |
| Cognitive scoring | -- | -- | -- | ACT-R + FadeMem | ACT-R + FadeMem + **surprise + fan effect** |
| Governance hierarchy | -- | -- | -- | Protected flag | **4-tier constitutional** (decay/archive per layer) |
| Hallucination detection | -- | -- | -- | -- | **HaluMem** (confidence decay + source verification) |
| Memory consolidation | -- | -- | -- | -- | **4-phase sleep** (replay/reorganize/compress/prune) |
| Intake filtering | -- | -- | Partial | Dedup only | **6-gate sensory filter** |
| Self-linking | -- | -- | -- | -- | **Zettelkasten** (auto bidirectional, cosine > 0.5) |
| Surprise scoring | -- | -- | -- | -- | **Titans-inspired** (centroid distance) |
| Structure awareness | -- | -- | -- | -- | **4 types** (timeline/ledger/taxonomy/narrative) |
| Fan effect | -- | -- | -- | -- | **ACT-R IDF penalty** |
| Multi-agent sync | -- | Partial | -- | Changelog | Changelog + **conflict resolution** |
| Async writes | -- | -- | -- | -- | **Priority queue** (1-5, batch flush) |
| MCP tools | -- | -- | -- | 17 | **24** |
| Embedding model | OpenAI | OpenAI | OpenAI | Local (Ollama) | Local (Ollama, **768-dim nomic-embed-text**) |
| LLM for extraction | External API | External API | External API | Local (Ollama) | Local (Ollama, **qwen2.5:3b**) |
| Privacy | -- | -- | -- | Full local | **Full local** (zero cloud dependency) |

---

## Abstract

memory-v3 is a next-generation persistent memory system for AI coding assistants,
implemented as a Model Context Protocol (MCP) server exposing 24 tools across
CRUD, search, graph traversal, lifecycle management, and system operations. It
stores memories in a SQLite database augmented with sqlite-vec for 768-dimensional
vector search and FTS5 for keyword retrieval, backed by four independent NetworkX
directed graphs (semantic, temporal, causal, entity) that together form the
Multi-Agent Graph Memory Architecture (MAGMA). Retrieval follows a 3-stage
pipeline: coarse candidate filtering via community routing, vector search, and
FTS5 (capped at 500 candidates); fine-grained scoring via Reciprocal Rank Fusion,
ACT-R cognitive activation with fan effect penalties, HaluMem confidence
adjustment, and Titans-inspired surprise weighting; and post-retrieval
organization using graph neighbor expansion, structure-type detection, and
zettelkasten see-also linking. A 4-tier constitutional governance hierarchy
(Constitutional, Legislative, Factual, Ephemeral) assigns per-layer decay rates,
activation floors, and archival policies, ensuring that identity-critical memories
never decay while ephemeral episodes are aggressively reclaimed. The system runs
entirely locally via Ollama for both embeddings (nomic-embed-text) and LLM
extraction (qwen2.5:3b), requiring zero cloud API keys, and totals approximately
10,565 lines of Python across 40 modules.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Theoretical Foundations](#theoretical-foundations)
  - [ACT-R Cognitive Scoring](#act-r-cognitive-scoring)
  - [FadeMem with Surprise](#fademem-with-surprise)
  - [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
  - [Unified Scoring Formula](#unified-scoring-formula)
  - [Fan Effect Penalty](#fan-effect-penalty)
  - [Surprise Scoring](#surprise-scoring)
  - [Governance Hierarchy](#governance-hierarchy)
- [3-Stage Retrieval Pipeline](#3-stage-retrieval-pipeline)
- [Multi-Graph Architecture (MAGMA)](#multi-graph-architecture-magma)
- [Database Schema](#database-schema)
- [MCP Server and Tools](#mcp-server-and-tools)
- [Subsystem Reference](#subsystem-reference)
- [Worked Examples](#worked-examples)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Lineage](#lineage)
- [Related Work](#related-work)
- [References](#references)

---

## Architecture Overview

```
                          +---------------------+
                          |    MCP Client        |
                          | (Claude Code, etc.)  |
                          +----------+----------+
                                     |
                                     | FastMCP (stdio/SSE)
                                     v
                     +-------------------------------+
                     |        server.py               |
                     |    24 MCP Tool Endpoints       |
                     +------+-------+-------+--------+
                            |       |       |
              +-------------+   +---+---+   +-------------+
              |                 |       |                   |
              v                 v       v                   v
    +------------------+  +---------+ +----------+  +---------------+
    |   Sensory Gate   |  |  CRUD   | |  Search  |  |   Lifecycle   |
    | (6-gate filter)  |  | (db.py) | | Pipeline |  |   Manager     |
    +--------+---------+  +----+----+ +----+-----+  +-------+-------+
             |                 |            |                |
             v                 v            v                v
    +------------------+  +--------+  +-----------+  +---------------+
    | credential scan  |  | SQLite |  | 3-Stage   |  | extraction    |
    | exact dedup      |  | + vec0 |  | Retrieval |  | consolidation |
    | near dedup       |  | + FTS5 |  |           |  | compaction    |
    | surprise score   |  | + WAL  |  | Stage 1:  |  | hallucination |
    | topic cluster    |  +---+----+  |  Coarse   |  | decay sweep   |
    | length check     |      |       | Stage 2:  |  +---------------+
    +------------------+      |       |  Fine     |
                              |       | Stage 3:  |
                              |       |  Organize |
                              |       +-----------+
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
    +------------------+ +---------+  +----------------+
    |  Scoring Engine  | |  Async  |  | MAGMA Graphs   |
    |                  | |  Write  |  |                |
    | ACT-R activation | |  Queue  |  | semantic.pkl   |
    | FadeMem decay    | | (1-5    |  | temporal.pkl   |
    | surprise score   | | priority|  | causal.pkl     |
    | fan effect       | | levels) |  | entity.pkl     |
    | hierarchy rules  | +---------+  |                |
    +------------------+              | Leiden detect  |
                                      | PPR traversal  |
                                      | Zettelkasten   |
                                      +----------------+
```

---

## Theoretical Foundations

### ACT-R Cognitive Scoring

memory-v3 implements the Adaptive Control of Thought -- Rational (ACT-R) framework
(Anderson & Lebiere, 1998) for computing memory activation levels. Higher activation
means a memory is more likely to be needed in the current context.

**Base-level activation** reflects how frequently and recently a memory has been
accessed:

```
B_i = ln(n / (1 - d)) - d * ln(L)
```

Where:
- `n` = access count (number of times the memory has been retrieved)
- `d` = decay parameter (default 0.5)
- `L` = lifetime in hours since memory creation

**Spreading activation** models contextual priming from shared tags:

```
S_i = SUM over shared tags j of: W_j * (S_max - ln(fan_j))
```

Where:
- `W_j = 1 / |context_tags|` (source activation distributed evenly)
- `S_max` = maximum associative strength (default 1.6)
- `fan_j` = number of memories associated with tag j

**Activation noise** adds stochastic variability via a logistic distribution:

```
epsilon ~ Logistic(0, s)

Sampled as: epsilon = s * ln(u / (1 - u))
    where u ~ Uniform(0, 1)
```

The logistic distribution has variance `(pi * s)^2 / 3`, with `s = 0.25` by default.

**Full activation** combines all components with a governance-aware floor:

```
A_i = max(B_i + S_i + epsilon, activation_floor)
```

Where `activation_floor` depends on the governance layer:
- Layer 1 (Constitutional): floor = 5.0
- Layer 2 (Legislative): floor = 2.0
- Layer 3 (Factual): floor = -0.5
- Layer 4 (Ephemeral): floor = -2.0

Protected memories receive an activation floor of at least 0.0.

**Retrieval probability** uses the sigmoidal (softmax) rule:

```
P_i = 1 / (1 + exp(-(A_i - tau) / s))
```

Where:
- `tau` = retrieval threshold (default -0.5)
- `s` = noise scale (default 0.25)


### FadeMem with Surprise

FadeMem manages memory lifecycle through a 4-weight importance function that
balances relevance, frequency, recency, and novelty. The v3 extension adds
a surprise component inspired by Titans (Google, 2024).

**Importance score:**

```
I(t) = alpha * relevance + beta * frequency + gamma * recency + delta * surprise
```

With default weights:
- `alpha = 0.30` (relevance -- semantic similarity to active context)
- `beta  = 0.25` (frequency -- log-normalized access count)
- `gamma = 0.25` (recency -- exponential decay from last access)
- `delta = 0.20` (surprise -- Titans-inspired novelty signal)

**Frequency** is log-normalized against the corpus maximum:

```
frequency = ln(access_count + 1) / ln(max_access_count + 1)
```

**Recency** uses exponential decay with a layer-aware rate:

```
recency = exp(-lambda * t)
```

Where `lambda` is the governance layer's decay rate and `t` is hours since last access.

**Weibull-style strength decay** for long-term value tracking:

```
v(t) = v(0) * exp(-lambda * t^beta)
```

Where `beta < 1` produces fast-then-slow decay and `beta > 1` produces slow-then-fast.

**Memory layer classification** based on importance:

```
importance >= 0.7  -->  LTM (long-term memory)
importance >= 0.3  -->  STM (short-term memory)
importance <  0.3  -->  current (candidate for archive)
```


### Reciprocal Rank Fusion

Stage 2 fuses BM25 keyword rankings and vector similarity rankings using RRF
(Cormack et al., 2009):

```
RRF(d) = 1 / (k + rank_bm25(d)) + 1 / (k + rank_vec(d))
```

Where:
- `k = 60` (smoothing constant, configurable via `MEMORY_V3_RRF_K`)
- `rank_bm25(d)` = 0-based position in FTS5 results (9999 if absent)
- `rank_vec(d)` = 0-based position in vector results (9999 if absent)

RRF is robust to score-scale mismatch between BM25 and cosine distance,
requiring no normalization.


### Unified Scoring Formula

The final retrieval score combines search relevance, cognitive activation,
and verification confidence:

```
final = 0.60 * hybrid_score + 0.25 * (A / 10.0) + 0.15 * confidence
```

Where:
- `hybrid_score` = RRF fusion score from BM25 + vector search
- `A` = full ACT-R activation (base-level + spreading + noise, floored)
- `confidence` = HaluMem-adjusted confidence (verification-aware)

**HaluMem confidence adjustment:**
- `verified` status: confidence unchanged
- `stale` status (> 90 days unverified): `confidence *= 0.7`
- `contradicted` status: `confidence *= 0.3`

The 60/25/15 weighting ensures search relevance dominates while cognitive
signals and verification status provide meaningful re-ranking.


### Fan Effect Penalty

The fan effect (Anderson, 1974) models the cognitive finding that as a concept
becomes associated with more facts, each individual association becomes harder
to retrieve. memory-v3 adds a logarithmic penalty to spreading activation:

```
fan_penalty(n) = delta * ln(n + 1)
```

Where `delta = 0.3` (configurable via `MEMORY_V3_FAN_DELTA`).

**Spreading activation with fan penalty:**

```
S_i = SUM over shared tags j of:
    W_j * max(S_max - ln(fan_j) - fan_penalty(total_connections_j), 0)
```

The per-tag contribution is floored at zero to prevent negative activation.
This is the ACT-R equivalent of IDF weighting: tags shared across many memories
contribute less spreading activation than rare, specific tags.


### Surprise Scoring

Surprise quantifies how unexpected a new memory is relative to the existing
knowledge base. Inspired by the Titans architecture (Google DeepMind, 2024),
surprise biases the system toward retaining novel information.

**Centroid-based surprise:**

```
surprise = 1.0 - max(cosine_similarity(embedding, c_i) for c_i in centroids)
```

Where centroids are the mean embeddings of each topic cluster, stored in the
`cluster_centroids` table and recomputed during consolidation.

- `surprise = 0.0` means the memory is identical to a known cluster center
- `surprise = 1.0` means the memory is maximally dissimilar to all clusters
- `surprise = 0.5` is the neutral default when no centroids exist

**Cluster assignment** maps each memory to its nearest centroid:

```
cluster(m) = argmin_i (1 - cosine_similarity(embedding_m, c_i))
```

Centroids are L2-normalized after recomputation for consistent cosine comparisons.


### Governance Hierarchy

The Constitutional Governance Hierarchy assigns every memory to one of four
tiers that control its decay behavior, activation floor, and archival eligibility:

| Layer | Name | Decay Rate | Activation Floor | FadeMem Beta | Archivable | Archive After | Example Types |
|---|---|---|---|---|---|---|---|
| 1 | Constitutional | 0.0 | 5.0 | N/A | Never | N/A | identity, chain_of_command |
| 2 | Legislative | 0.01 | 2.0 | 0.5 | Never | N/A | decision, correction, commitment |
| 3 | Factual | 0.10 | -0.5 | 0.8 | Yes | 30 days, I < 0.1 | fact, reference, tool |
| 4 | Ephemeral | 0.25 | -2.0 | 1.2 | Yes | 14 days, I < 0.15 | episode, conversation |

**Auto-classification priority:**

1. Tag-based override (identity/chain_of_command tags -> Layer 1; decision/correction/commitment/exact_value tags -> Layer 2)
2. Content type mapping (content_type string to layer)
3. Protected flag bump (protected memories get at least Layer 2)
4. Default to Layer 3 (Factual)

**Archival rules:**

```
should_archive(m) =
    NOT (tags INTERSECT protected_tags) AND
    layer.can_archive AND
    importance < layer.archive_threshold AND
    age_days > layer.min_archive_age_days
```

Protected tags that always prevent archival: `correction`, `decision`, `identity`,
`emotional_anchor`, `commitment`, `exact_value`, `chain_of_command`, `person`.

---

## 3-Stage Retrieval Pipeline

### Stage 1: Coarse Candidate Filtering

Fast, broad retrieval to build a candidate pool. Three channels run
in parallel:

1. **Community routing** -- Find top-N communities by centroid similarity
   to the query embedding, then collect all memory IDs in those communities.
   Default `N = 3` (configurable via `MEMORY_V3_STAGE1_COMMUNITY_TOP`).

2. **Vector search** -- Fast approximate nearest neighbor via the `memory_vec`
   virtual table (vec0), returning up to 200 candidates ordered by distance.

3. **FTS5 keyword search** -- Full-text search with per-word quoting for
   safe special character handling, returning up to 200 candidates by rank.

The three channels are unioned with ordered deduplication (vector results first,
then FTS, then community), capped at `stage1_max_candidates` (default 500).

### Stage 2: Fine-Grained Scoring

Each candidate from Stage 1 receives a composite score:

1. **Re-rank via RRF** -- BM25 and vector rank positions are fused:
   `rrf = 1/(k + bm25_rank) + 1/(k + vec_rank)`

2. **ACT-R activation** -- Base-level + spreading (with fan penalty) + noise,
   floored by governance layer.

3. **HaluMem confidence** -- Verification status adjusts confidence:
   stale -> 0.7x, contradicted -> 0.3x.

4. **Final score** -- `0.60 * rrf + 0.25 * (A / 10) + 0.15 * confidence`

Results are sorted by final score descending.

### Stage 3: Post-Retrieval Organization

Scored results are organized for LLM consumption:

1. **Structure detection** -- Majority vote across results determines the
   dominant structure type (timeline, ledger, taxonomy, narrative).

2. **Query routing** -- The query is routed to relevant graph layers
   (semantic, temporal, causal, entity) based on keyword patterns.

3. **Structure-appropriate ordering:**
   - Timeline or temporal route -> chronological sort
   - Causal route -> causal chain ordering (root cause -> effect)
   - Entity route -> grouped by entity associations
   - Narrative/default -> score-based ordering from Stage 2

4. **Graph neighbor expansion** -- Each result is annotated with 1-hop
   neighbors from relevant graph layers (up to 10 per result).

5. **Zettelkasten see-also** -- Each result receives up to 5 linked memory
   previews based on cosine similarity links.

6. **Structure hint** -- A plain-English hint is attached to guide LLM
   interpretation (e.g., "These are timeline entries, presented in
   chronological order. (12 entries returned.)").

---

## Multi-Graph Architecture (MAGMA)

MAGMA maintains four independent NetworkX DiGraph instances, each specialized
for a different relationship type:

### Semantic Graph

Models conceptual relationships between entities, tools, and systems.

**Edge types:** `related_to`, `uses`, `part_of`, `implements`, `depends_on`, `similar_to`

**Use case:** "What technologies relate to this project?" traverses `uses` and
`depends_on` edges to surface the dependency tree.

### Temporal Graph

Tracks chronological relationships between events and sessions.

**Edge types:** `preceded_by`, `followed_by`, `concurrent_with`, `overlaps_with`, `same_session`

**Use case:** "What happened before the deploy failure?" follows `preceded_by`
edges to reconstruct the event timeline.

### Causal Graph

Encodes cause-and-effect relationships between events and decisions.

**Edge types:** `caused`, `enabled`, `prevented`, `motivated`, `triggered_by`, `resolved`

**Use case:** "Why did the tests break?" traces `caused` and `triggered_by`
edges from the symptom to the root cause.

### Entity Graph

Maps person/organization/system relationships and responsibilities.

**Edge types:** `built`, `decided`, `maintains`, `member_of`, `responsible_for`

**Use case:** "Who is responsible for the auth service?" follows
`responsible_for` and `maintains` edges.

### Graph Operations

All four layers share common operations from the base graph class:

- **Personalized PageRank** -- Biased random walk from seed nodes, with
  configurable `alpha` (teleport probability) and `top_k` results.
- **Leiden community detection** -- Via python-igraph and leidenalg (optional
  dependency), with community IDs stored on each node.
- **Persistence** -- Each graph is pickle-serialized to `~/.memory-v3/graphs/`.

### Query Routing

The `router.py` module maps query keywords to relevant graph layers:

```python
route_query("what happened before the deploy?")
# -> ["temporal", "causal"]

route_query("who maintains the auth service?")
# -> ["entity", "semantic"]
```

---

## Database Schema

memory-v3 uses SQLite with WAL mode, sqlite-vec for vector search, and FTS5
for keyword search. The schema contains 14 tables:

### Core Tables

**memories** -- Primary storage for all memories.

| Column | Type | Default | Description |
|---|---|---|---|
| id | INTEGER | auto | Primary key |
| content | TEXT | required | Memory content |
| content_type | TEXT | 'fact' | fact/episode/decision/correction/identity/person |
| source_file | TEXT | NULL | Originating file path |
| source_line | INTEGER | NULL | Line number in source |
| author | TEXT | 'claude-code' | Creating agent |
| authority_level | INTEGER | 2 | Agent authority (1=human, 6=extractor) |
| confidence | REAL | 0.95 | Belief confidence [0, 1] |
| protected | INTEGER | 0 | Boolean: never decay |
| tags | TEXT | '[]' | JSON array of tag strings |
| created_at | TEXT | required | ISO 8601 UTC timestamp |
| updated_at | TEXT | required | Last modification time |
| last_accessed_at | TEXT | required | Last retrieval time |
| access_count | INTEGER | 1 | Retrieval counter |
| activation_score | REAL | 0.0 | Cached ACT-R activation |
| importance_score | REAL | 0.5 | FadeMem importance |
| decay_rate | REAL | 0.1 | Per-memory decay rate |
| archived | INTEGER | 0 | Boolean: in graveyard |
| supersedes | INTEGER | NULL | ID of memory this replaces |
| content_hash | TEXT | NULL | SHA-256 prefix for dedup |
| governance_layer | INTEGER | 3 | Governance tier (1-4) |
| confidence_floor | REAL | 0.0 | Minimum confidence |
| last_verified_at | TEXT | NULL | Last HaluMem verification |
| verification_status | TEXT | 'unverified' | verified/stale/contradicted/unverified |
| cross_ref_count | INTEGER | 0 | Graph cross-references |
| structure_type | TEXT | 'narrative' | narrative/timeline/ledger/taxonomy |
| surprise_score | REAL | 0.5 | Titans novelty score |
| memory_layer | TEXT | 'STM' | STM/LTM/current |
| cluster_id | INTEGER | NULL | Topic cluster assignment |
| linked_memories | TEXT | '[]' | JSON array of linked IDs |
| link_descriptions | TEXT | '{}' | JSON map of link descriptions |
| source_hash | TEXT | NULL | Source file content hash |

**memory_vec** -- vec0 virtual table for 768-dim vector search.

```sql
CREATE VIRTUAL TABLE memory_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[768]
);
```

**memory_fts** -- FTS5 index over content, tags, and source_file.

```sql
CREATE VIRTUAL TABLE memory_fts USING fts5(
    content, tags, source_file,
    content='memories', content_rowid='id'
);
```

FTS5 is kept in sync via `AFTER INSERT`, `AFTER DELETE`, and `AFTER UPDATE`
triggers on the `memories` table.

### Lifecycle Tables

| Table | Purpose |
|---|---|
| graveyard | Archived memories with metadata and archival reason |
| sensory_buffer | Pre-encoding intake queue with content hash, novelty score, topic cluster |
| cluster_centroids | Mean embeddings per topic cluster (recomputed during consolidation) |
| write_queue | Async write operations with priority (1-5) and status tracking |
| action_log | Extraction decisions (ADD/UPDATE/DELETE/NONE) with confidence |
| consolidation_log | Sleep cycle records with per-phase stats |
| compaction_receipts | CogCanvas compression records with ratio and verification score |

### Infrastructure Tables

| Table | Purpose |
|---|---|
| agent_offsets | Multi-agent sync: last-read position per agent |
| file_hashes | Vault file hash registry for incremental indexing |
| conflicts | Inter-agent memory conflicts with resolution tracking |
| schema_version | Migration version tracking |

### Indexes

```sql
-- v2 indexes (content access patterns)
idx_memories_type, idx_memories_archived, idx_memories_protected,
idx_memories_source, idx_memories_created, idx_memories_activation

-- v3 indexes (governance and lifecycle)
idx_memories_governance, idx_memories_memory_layer, idx_memories_cluster,
idx_memories_verification, idx_sensory_processed, idx_write_queue_status,
idx_action_log_action
```

---

## MCP Server and Tools

memory-v3 exposes 24 tools via FastMCP, organized into five categories.

### CRUD (5 tools)

| Tool | Description |
|---|---|
| `add_memory` | Store a new memory with auto-classification (governance layer, structure type, surprise score). Routes through credential scan, embedding, novelty check, and optional async write queue. Zettelkasten auto-linking runs in a background thread. |
| `get` | Retrieve a single memory by ID with full content, metadata, governance info, and verification status. |
| `update` | Modify an existing memory's content, tags, protected status, governance layer, or structure type. |
| `forget` | Archive a memory to the graveyard. Protected memories and Layer 1/2 memories are rejected. |
| `link_memories` | Create a manual bidirectional zettelkasten link between two memories with an optional description. |

### Search (5 tools)

| Tool | Description |
|---|---|
| `search` | Full 3-stage retrieval pipeline: coarse filtering (community + vec0 + FTS5) -> fine scoring (RRF + ACT-R + fan effect + HaluMem) -> organization (graph neighbors + structure hints + zettelkasten see-also). Returns ranked results with scores. |
| `keyword_search` | Pure BM25 keyword search via FTS5. Use for exact terms, file names, error codes. |
| `graph_search` | Knowledge graph search using Personalized PageRank. Discovers related concepts via multi-hop graph traversal across all four MAGMA layers. |
| `get_causal_chain` | Trace cause-and-effect chains in the causal graph. Returns root-cause-to-effect paths with edge types. |
| `get_timeline` | Query the temporal graph for chronologically ordered events. Supports optional date range filtering. |

### System (5 tools)

| Tool | Description |
|---|---|
| `stats` | System-wide statistics: memory counts by type/layer/status, write queue depth, sensory buffer size, graph node/edge counts. |
| `list_recent` | List memories created in the last N hours (default 24), with content preview, type, tags, and governance layer. |
| `list_topics` | List topic clusters from graph communities (connected components) or content type distribution as fallback. |
| `reindex` | Re-index the vault directory. Incremental by default (only changed files via SHA-256 hash comparison). Detects content type, extracts tags, classifies governance layer. |
| `check_integrity` | Vault file integrity check against saved manifest, plus staleness detection (memories > 90 days unverified), plus SQLite `PRAGMA integrity_check`. |

### Lifecycle (5 tools)

| Tool | Description |
|---|---|
| `extract_from_conversation` | 2-pass LLM extraction pipeline: extract candidate facts from text, then decide ADD/UPDATE/DELETE/NONE for each. Routes through sensory filter first if enabled. |
| `compact_text` | CogCanvas text compression: extract protected content, delete expendable material, summarize remainder, verify faithfulness. Returns compressed text with receipt. |
| `decay_sweep` | FadeMem maintenance: recalculate importance scores with layer-aware decay rates, promote/demote between STM/LTM, archive low-value memories respecting governance, decay confidence for stale memories via HaluMem. |
| `sleep_consolidation` | 4-phase sleep cycle: REPLAY (zettelkasten links for last 24h), REORGANIZE (update centroids and communities), COMPRESS (merge similar memories > 0.85 cosine), PRUNE (archive below governance thresholds). Accepts phase selection or 'full'. |
| `verify_memories` | HaluMem batch verification: check source files exist and content matches, update confidence and verification_status (verified/stale/contradicted/unverified). |

### Advanced (4 tools)

| Tool | Description |
|---|---|
| `agent_sync` | Multi-agent synchronization: read unprocessed changelog entries for a specific agent and advance its offset. |
| `check_conflicts` | List unresolved inter-agent memory conflicts with agent IDs, memory IDs, and descriptions. |
| `set_governance_layer` | Change a memory's governance tier (1-4). Layer 1 assignment requires human authority level. |
| `list_topics` | List topic clusters from graph communities or content type distribution. |

---

## Durable facts and temporary handoffs

For a lasting fact, edit its canonical vault/project document, run `memory-v3 reindex`
(or the MCP `reindex` tool), then use `search` or `keyword_search` to verify the
new text is retrieved with its source path. The index is derived from documents;
direct `add_memory` and `extract_from_conversation` writes create expiring handoffs
only. `durable=True`, admin overrides, and custom write namespaces are rejected.
Do not use a handoff as a substitute for updating the source document.

## Subsystem Reference

### Sensory Gate (lifecycle/sensory.py)

The sensory gate is a 6-stage intake filter inspired by LightMem that runs
BEFORE any LLM calls, rejecting noise at minimal compute cost.

**Gate 1 -- Length check:** Content shorter than `sensory_min_length` (default 20
characters) is rejected.

**Gate 2 -- Credential scan:** Regex-based detection of API keys, tokens, passwords,
and other secrets. Matches reject the content immediately.

**Gate 3 -- Exact dedup (SHA-256):** Content hash is checked against both the
sensory buffer and the memories table. Exact matches are rejected.

**Gate 4 -- Near-dedup (cosine > 0.95):** The content is embedded and compared
against the last 100 sensory buffer entries and the top 5 nearest vectors in
`memory_vec`. Cosine similarity exceeding `sensory_dedup_threshold` (default 0.95)
triggers rejection.

**Gate 5 -- Surprise scoring:** Computes centroid-based surprise. Low-surprise
content is accepted but flagged; the score is carried forward to influence
FadeMem importance.

**Gate 6 -- Topic clustering:** Assigns the content to its nearest cluster
centroid. This assignment propagates to the memory's `cluster_id` column.

### HaluMem Hallucination Detection (lifecycle/hallucination.py)

HaluMem addresses the problem of unverified memories accumulating false
confidence over time.

**Confidence decay:** Memories not verified within `halumem_confidence_decay_days`
(default 90) have their confidence reduced by `halumem_confidence_decay_rate`
(default 0.1) per sweep. The `verification_status` field transitions:
`unverified -> stale -> contradicted`.

**Batch verification:** Checks whether source files still exist and whether their
content still matches the memory's `source_hash`. Updates `verification_status`
and `last_verified_at` accordingly.

**Staleness detection:** Queries for memories exceeding the staleness threshold,
used by `check_integrity` to surface memories needing re-verification.

### Zettelkasten Self-Linking (graphs/zettelkasten.py)

Inspired by Niklas Luhmann's slip-box method, the zettelkasten module creates
bidirectional links between semantically similar memories.

**Auto-linking:** When a new memory is added, its embedding is compared against all
existing memories. Pairs exceeding `zettel_link_threshold` (default 0.5 cosine
similarity) are linked bidirectionally. Maximum 10 links per memory.

**Background threading:** Auto-link runs in a daemon thread to avoid blocking the
`add_memory` response.

**Link retrieval:** The `get_links` function returns all linked memories for a given
ID, with similarity scores and content previews. These appear as `see_also`
annotations in Stage 3 results.

### Sleep Consolidation (lifecycle/consolidation.py)

Four-phase consolidation inspired by biological memory consolidation during sleep:

**Phase 1 -- REPLAY:** Re-process memories created in the last 24 hours. Run
zettelkasten auto-link for each, creating cross-connections that were not
established at ingestion time.

**Phase 2 -- REORGANIZE:** Recompute cluster centroids as the mean embedding of
their members. Update community assignments via connected component analysis
(or Leiden community detection if `python-igraph` and `leidenalg` are installed).

**Phase 3 -- COMPRESS:** Find groups of 3+ memories with pairwise cosine similarity
exceeding `consolidation_merge_threshold` (default 0.85). Merge each group into
a single summary memory, archiving the originals. The merged memory inherits the
highest governance layer and protected status from the group.

**Phase 4 -- PRUNE:** Iterate all active memories, recalculate importance, and
archive those below their governance layer's threshold. Respects protected tags
and Layer 1/2 immunity.

### Async Write Queue (async_ops/write_queue.py)

The write queue decouples memory creation from the MCP response path:

- **Priority levels 1-5:** Layer 1 (Constitutional) memories get priority 1;
  Layer 4 (Ephemeral) get priority 4. Priority 5 is reserved for background
  operations.
- **Batch flushing:** The queue processes operations in priority order, with
  configurable batch size (default 10).
- **Failure persistence:** Failed operations remain in the queue with error
  details for retry.
- **Immediate flush:** Despite async architecture, `add_memory` flushes
  immediately after enqueue for sync-like behavior. True async is used for
  background indexing and consolidation.

### Structure Tags (retrieval/structure_tags.py)

Structure tags classify memory content into four organizational types:

| Type | Description | Sort Strategy | Hint |
|---|---|---|---|
| timeline | Chronologically ordered events | created_at ascending | "These are timeline entries, presented in chronological order." |
| ledger | Key-value pairs (config, env vars) | None (score order) | "These are ledger entries (key-value records). Each is a standalone fact." |
| taxonomy | Hierarchical classification | None (score order) | "These are taxonomy entries organized hierarchically." |
| narrative | Story-form content | created_at ascending | "These are narrative entries. They tell a story in sequence." |

**Auto-detection** uses regex pattern matching:
- Ledger: `key: value` or `key = value` patterns (>= 2 matches)
- Timeline: date patterns (`YYYY-MM-DD`), sequence words (`first`, `then`, `next`)
- Taxonomy: indented bullets, hierarchy keywords (`parent`, `child`, `inherits`)
- Narrative: default (base score of 1)

---

## Worked Examples

### Example 1: Adding a Memory and Scoring

A user stores: "The auth service uses JWT tokens with RS256 signing, rotated every 90 days."

**Step 1 -- Sensory gate:**
- Length: 73 chars > 20 (pass)
- Credential scan: no matches (pass)
- Exact dedup: SHA-256 not in buffer or memories (pass)
- Near dedup: closest cosine = 0.42 < 0.95 (pass)
- Surprise: centroid distance = 0.67 (moderately novel)
- Cluster: assigned to cluster 3 (security)

**Step 2 -- Classification:**
- Content type: "fact"
- Governance: Layer 3 (Factual) -- no protected tags, no override
- Structure: "ledger" -- matches `key: value` pattern
- Surprise: 0.67

**Step 3 -- Importance:**
```
importance = 0.5 + (0.67 * 0.2) + (0.95 * 0.1) = 0.5 + 0.134 + 0.095 = 0.729
```

**Step 4 -- Zettelkasten (background):**
- Compared against all memories
- Linked to memory #42 ("JWT configuration") with cosine 0.78
- Linked to memory #105 ("RS256 key rotation policy") with cosine 0.63

### Example 2: 3-Stage Retrieval

Query: "How does authentication work?"

**Stage 1 -- Coarse filtering (produces 87 candidates):**
- Community routing: top-3 communities -> 23 memory IDs
- Vector search (top 200): 52 unique IDs
- FTS5 for "How" OR "does" OR "authentication" OR "work": 41 unique IDs
- Union (vec first, then FTS, then community): 87 unique candidates

**Stage 2 -- Fine scoring (top candidate):**
- Memory #42: "Auth service configuration: JWT with RS256..."
  ```
  BM25 rank: 2    ->  1/(60+2) = 0.01613
  Vector rank: 0   ->  1/(60+0) = 0.01667
  RRF score: 0.03280

  Base-level: B = ln(12 / 0.5) - 0.5 * ln(720) = 3.178 - 3.288 = -0.110
  Spreading:  S = 0.5 * (1.6 - ln(8)) = 0.5 * (1.6 - 2.079) = -0.240
  Fan penalty:  0.3 * ln(9) = 0.659
  Adjusted S: 0.5 * max(1.6 - 2.079 - 0.659, 0) = 0.0
  Noise:      epsilon = -0.03 (sampled)
  Floor:      -0.5 (Layer 3)
  Activation: A = max(-0.110 + 0.0 + (-0.03), -0.5) = -0.140

  Confidence: 0.95 (verified status, no adjustment)

  Final: 0.60 * 0.03280 + 0.25 * (-0.140/10) + 0.15 * 0.95
       = 0.01968 + (-0.00350) + 0.14250
       = 0.15868
  ```

**Stage 3 -- Organization:**
- Dominant structure: "ledger" (60% of results)
- Routed layers: ["semantic"] (no temporal/causal keywords)
- Ordering: score-based (ledger uses default)
- Graph neighbors: `auth_service --uses--> JWT`, `JWT --implements--> RS256`
- Zettelkasten: see_also -> memory #105 (cosine 0.63)
- Hint: "These are ledger entries (key-value records). Each is a standalone fact. (87 entries returned.)"

### Example 3: Governance-Aware Decay Sweep

Three memories evaluated during `decay_sweep`:

**Memory A -- Layer 1 (Constitutional):**
- Content: "I am Claude, an AI assistant by Anthropic"
- Governance layer 1 -> skipped entirely (Constitutional never decays)

**Memory B -- Layer 3 (Factual), 45 days old:**
- importance before: 0.35
- Recalculated:
  ```
  frequency = ln(3+1) / ln(50+1) = 1.386 / 3.932 = 0.352
  recency   = exp(-0.10 * 720) = ~0.0 (essentially zero)
  surprise  = 0.45
  relevance = 0.4

  I = 0.30*0.4 + 0.25*0.352 + 0.25*0.0 + 0.20*0.45
    = 0.120 + 0.088 + 0.000 + 0.090 = 0.298
  ```
- New importance: 0.298 (changed by > 0.01, updated)
- Memory layer: 0.298 < 0.3 -> "current" (demoted from STM)
- Archive check: 0.298 > 0.1 threshold AND age 45 > 30 days... but 0.298 > 0.1, so NOT archived

**Memory C -- Layer 4 (Ephemeral), 20 days old:**
- importance: 0.08
- Archive check: 0.08 < 0.15 threshold AND 20 > 14 days -> ARCHIVED
- Moved to graveyard with reason "decay_sweep"

---

## Installation

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) running locally with:
  - `nomic-embed-text` (embedding model, 768 dimensions)
  - `qwen2.5:3b` (LLM for extraction and compaction)

```bash
# Install Ollama models
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

### Install from source

```bash
git clone https://github.com/Haustorium12/memory-v3.git
cd memory-v3
pip install -e ".[all]"
```

The `[all]` extra includes graph dependencies (`python-igraph`, `leidenalg`,
`pyvis`) and development tools (`pytest`, `ruff`).

### Install core only

```bash
pip install -e .
```

Core dependencies: `fastmcp>=2.0`, `numpy>=1.24`, `networkx>=3.0`,
`ollama>=0.4`, `sqlite-vec>=0.1`.

### Configure as MCP server

Add to your Claude Code `settings.json`:

```json
{
  "mcpServers": {
    "memory-v3": {
      "command": "memory-v3-server",
      "env": {
        "MEMORY_V3_DB": "/path/to/memory.db",
        "MEMORY_V3_VAULT": "/path/to/vault"
      }
    }
  }
}
```

Or with `uv`:

```json
{
  "mcpServers": {
    "memory-v3": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/memory-v3",
        "memory-v3-server"
      ],
      "env": {
        "MEMORY_V3_DB": "/path/to/memory.db"
      }
    }
  }
}
```

---

## Configuration

All configuration is via `MEMORY_V3_*` environment variables. The system reads
them once at startup and caches the result.

### Paths

| Variable | Default | Description |
|---|---|---|
| `MEMORY_V3_DB` | `~/.memory-v3/memory.db` | SQLite database path |
| `MEMORY_V3_VAULT` | (none) | Markdown vault directory for indexing |
| `MEMORY_V3_GRAPH_DIR` | `~/.memory-v3/graphs` | Graph pickle directory |
| `MEMORY_V3_CACHE` | `~/.memory-v3/cache` | Cache directory |

### Models

| Variable | Default | Description |
|---|---|---|
| `MEMORY_V3_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `MEMORY_V3_LLM_MODEL` | `qwen2.5:3b` | Ollama LLM for extraction |
| `MEMORY_V3_EMBED_DIM` | `768` | Embedding dimensionality |

### ACT-R Parameters

| Variable | Default | Description |
|---|---|---|
| `MEMORY_V3_ACTR_DECAY_D` | `0.5` | Base-level decay parameter |
| `MEMORY_V3_ACTR_S_MAX` | `1.6` | Maximum associative strength |
| `MEMORY_V3_ACTR_NOISE_S` | `0.25` | Logistic noise scale |
| `MEMORY_V3_ACTR_RETRIEVAL_TAU` | `-0.5` | Retrieval threshold |

### FadeMem Weights

| Variable | Default | Description |
|---|---|---|
| `MEMORY_V3_FADEMEM_ALPHA` | `0.30` | Relevance weight |
| `MEMORY_V3_FADEMEM_BETA` | `0.25` | Frequency weight |
| `MEMORY_V3_FADEMEM_GAMMA` | `0.25` | Recency weight |
| `MEMORY_V3_FADEMEM_DELTA` | `0.20` | Surprise weight |

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `MEMORY_V3_RRF_K` | `60` | RRF smoothing constant |
| `MEMORY_V3_STAGE1_MAX_CANDIDATES` | `500` | Stage 1 candidate cap |
| `MEMORY_V3_STAGE1_COMMUNITY_TOP` | `3` | Top communities for routing |
| `MEMORY_V3_ZETTEL_LINK_THRESHOLD` | `0.5` | Cosine threshold for auto-linking |

### Other

| Variable | Default | Description |
|---|---|---|
| `MEMORY_V3_FAN_DELTA` | `0.3` | Fan effect penalty coefficient |
| `MEMORY_V3_SENSORY_MIN_LENGTH` | `20` | Minimum content length (chars) |
| `MEMORY_V3_SENSORY_DEDUP_THRESHOLD` | `0.95` | Near-dedup cosine threshold |
| `MEMORY_V3_CONSOLIDATION_MERGE_THRESHOLD` | `0.85` | Compression merge threshold |
| `MEMORY_V3_CONSOLIDATION_MIN_GROUP_SIZE` | `3` | Minimum group size for merge |
| `MEMORY_V3_HALUMEM_CONFIDENCE_DECAY_DAYS` | `90` | Staleness threshold (days) |
| `MEMORY_V3_HALUMEM_CONFIDENCE_DECAY_RATE` | `0.1` | Confidence decay per sweep |

### Feature Flags

All feature flags follow the pattern `MEMORY_V3_ENABLE_*` and accept
`1/true/yes/on` or `0/false/no/off`.

| Flag | Default | Description |
|---|---|---|
| `ENABLE_MULTIGRAPH` | `true` | 4-layer MAGMA graphs |
| `ENABLE_SENSORY` | `true` | 6-gate sensory filter |
| `ENABLE_ZETTELKASTEN` | `true` | Auto bidirectional linking |
| `ENABLE_SURPRISE` | `true` | Titans surprise scoring |
| `ENABLE_FAN_EFFECT` | `true` | ACT-R fan penalty |
| `ENABLE_THREE_STAGE` | `true` | 3-stage retrieval pipeline |
| `ENABLE_HALUMEM` | `true` | Hallucination detection |
| `ENABLE_HIERARCHY` | `true` | Constitutional governance |
| `ENABLE_CONSOLIDATION` | `true` | Sleep consolidation |
| `ENABLE_ASYNC` | `true` | Async write queue |
| `ENABLE_STRUCTURE_TAGS` | `true` | Structure type detection |
| `ENABLE_RL_DECIDER` | `false` | Reinforcement learning action decider (experimental) |

---

## Usage

### CLI Commands

```bash
# Start the MCP server
memory-v3-server

# Build/rebuild knowledge graphs from existing memories
memory-v3-build-kg

# Migrate from memory-v2
memory-v3-migrate

# CLI interface
memory-v3 stats
memory-v3 search "authentication configuration"
memory-v3 add "The deploy key rotates every 30 days" --type fact --tags security,deploy
memory-v3 recent --hours 48
```

### Typical Agent Workflow

```
1. Boot:     agent_sync("claude-code")     -- catch up on changes
2. Recall:   search("current project")     -- get relevant context
3. Store:    add_memory(content, type, tags) -- save new knowledge
4. Extract:  extract_from_conversation(text) -- bulk extraction
5. Maintain: decay_sweep()                  -- periodic maintenance
6. Sleep:    sleep_consolidation("full")    -- nightly consolidation
7. Verify:   verify_memories()              -- check source freshness
```

---

## Repository Structure

```
memory-v3/
  pyproject.toml              -- Package config, dependencies, entry points
  README.md                   -- This file
  src/
    memory_v3/
      __init__.py             -- Package init, embedder factory
      server.py        (1122) -- FastMCP server, 24 tool endpoints
      db.py             (864) -- SQLite schema, CRUD, search, migrations
      config.py         (311) -- Environment config, governance layer definitions
      embeddings.py      (50) -- Nomic embed-text wrapper (Ollama)
      cli.py            (464) -- CLI commands (stats, search, add, recent)
      build_kg.py       (481) -- Knowledge graph construction from memories
      migration.py      (453) -- v2 -> v3 database migration
      multi_agent.py    (418) -- Multi-agent sync, conflict detection
      security.py       (151) -- Credential scanning (regex patterns)

      async_ops/                -- Asynchronous write infrastructure
        __init__.py
        write_queue.py  (275) -- Priority queue (1-5), batch flush, retry
        batch_embedder.py(112) -- Batch embedding for bulk operations

      graphs/                   -- MAGMA: 4-layer knowledge graph
        __init__.py     (277) -- GraphManager, edge-to-layer routing
        base.py         (181) -- Base graph ops, PPR, persistence
        semantic.py     (178) -- Conceptual relationships
        temporal.py     (165) -- Chronological relationships
        causal.py       (196) -- Cause-and-effect chains
        entity.py       (174) -- Person/org/system relationships
        zettelkasten.py (292) -- Auto bidirectional linking
        communities.py  (223) -- Leiden detection, centroid routing
        router.py        (70) -- Query -> graph layer routing

      lifecycle/                -- Memory lifecycle management
        __init__.py
        extraction.py   (466) -- 2-pass LLM extraction pipeline
        consolidation.py(458) -- 4-phase sleep (replay/reorganize/compress/prune)
        compaction.py   (411) -- CogCanvas text compression
        sensory.py      (186) -- 6-gate intake filter
        hallucination.py(305) -- HaluMem confidence decay and verification
        action_decider.py(340) -- ADD/UPDATE/DELETE/NONE decision logic

      retrieval/                -- 3-stage retrieval pipeline
        __init__.py     (102) -- Pipeline entry point
        stage1_coarse.py(191) -- Community + vec0 + FTS5 candidate pool
        stage2_fine.py  (251) -- RRF + ACT-R + fan + confidence scoring
        stage3_organize.py(327) -- Graph context, structure hints, see-also
        structure_tags.py(104) -- Type detection and LLM hint generation

      scoring/                  -- Cognitive scoring engine
        __init__.py     (192) -- Unified score_memory() and rank_results()
        actr.py         (158) -- ACT-R base-level, spreading, noise, retrieval P
        fademem.py      (152) -- Importance, Weibull decay, archive decisions
        surprise.py     (155) -- Centroid-based surprise, cluster assignment
        hierarchy.py    (135) -- Governance layer configs and auto-classification
        fan_effect.py    (61) -- Spreading activation with fan penalty

  tests/
    conftest.py               -- Shared fixtures (in-memory DB, config)
    test_db.py                -- Database CRUD and schema tests
    test_migration.py         -- v2 -> v3 migration tests
    test_security.py          -- Credential scanning tests
    test_scoring/
      test_actr.py            -- ACT-R activation and retrieval probability
      test_fademem.py         -- Importance scoring and archival logic
      test_surprise.py        -- Centroid surprise and cluster assignment
      test_hierarchy.py       -- Governance layer classification
    test_graphs/
      test_router.py          -- Query routing to graph layers
```

Line counts in parentheses. Total: ~10,565 lines across 40 Python modules.

---

## Lineage

memory-v3 is the third generation of the memory system:

| Version | Repository | Architecture | Key Innovation |
|---|---|---|---|
| v1 | [claude-memory](https://github.com/Haustorium12/claude-memory) | Flat-file markdown | Persistent memory for Claude Code |
| v2 | [memory-v2](https://github.com/Haustorium12/memory-v2) | Single knowledge graph, ACT-R, hybrid search | Cognitive scoring, graph traversal |
| **v3** | **[memory-v3](https://github.com/Haustorium12/memory-v3)** (this repo) | Multi-graph, 3-stage retrieval, 12 upgrades | Constitutional hierarchy, MAGMA, sensory gate, sleep consolidation |

### 12 Upgrades from v2 to v3

1. **Constitutional Governance Hierarchy** -- 4-tier system replacing the boolean `protected` flag
2. **Multi-Graph Architecture (MAGMA)** -- 4 specialized DiGraphs replacing single graph
3. **3-Stage Retrieval Pipeline** -- Coarse/fine/organize replacing flat hybrid search
4. **ACT-R Cognitive Scoring** -- Enhanced with fan effect and governance-aware floors
5. **FadeMem with Surprise** -- 4-weight importance with Titans-inspired novelty signal
6. **HaluMem Hallucination Detection** -- Confidence decay for unverified memories
7. **Zettelkasten Self-Linking** -- Auto bidirectional links via cosine similarity
8. **Sensory Gate** -- 6-gate intake filter before any LLM processing
9. **Sleep Consolidation** -- 4-phase biological memory consolidation
10. **Async Write Queue** -- Priority-based batched writes
11. **Fan Effect** -- ACT-R IDF-equivalent penalty on overloaded tags
12. **Structure Tags** -- Content-aware organization for LLM consumption

---

## Related Work

**ACT-R** (Anderson & Lebiere, 1998) -- The Adaptive Control of Thought --
Rational cognitive architecture provides the base-level activation equation
and spreading activation mechanism used in memory-v3's scoring engine.

**Titans** (Google DeepMind, 2024) -- The surprise-based memory retention
mechanism from the Titans architecture inspires memory-v3's novelty scoring,
where memories maximally distant from known cluster centroids receive
retention priority.

**LightMem** -- The sensory gate architecture draws from LightMem's intake
filtering approach, implementing multiple rejection gates before expensive
LLM processing.

**CogCanvas** -- The compaction pipeline (extract protected content, delete
expendable material, summarize remainder, verify faithfulness) follows the
CogCanvas compression methodology.

**FadeMem** -- The importance-weighted decay function extends the FadeMem
framework with a fourth weight (surprise/novelty) and governance-aware
decay rates.

**Reciprocal Rank Fusion** (Cormack et al., 2009) -- RRF provides a
parameter-free fusion of BM25 and vector search rankings, requiring no
score normalization between the two retrieval channels.

**Zettelkasten** (Luhmann, 1981) -- Niklas Luhmann's slip-box method of
bidirectional note linking inspires the auto-linking subsystem, where
memories above a cosine similarity threshold are connected as "see also"
references.

**Leiden Algorithm** (Traag et al., 2019) -- Community detection in the
knowledge graphs uses the Leiden algorithm (when python-igraph is installed)
for identifying topic clusters that guide Stage 1 retrieval routing.

**sqlite-vec** -- Virtual table extension for approximate nearest neighbor
search in SQLite, enabling vector retrieval without external services.

**MCP (Model Context Protocol)** -- Anthropic's protocol for tool
integration between LLMs and external systems, providing the transport
layer for all 24 memory-v3 tools.

---

## References

1. Anderson, J. R., & Lebiere, C. (1998). *The Atomic Components of Thought.* Lawrence Erlbaum Associates.
2. Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology*, 6(4), 451-474.
3. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR '09*.
4. Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9(1), 5233.
5. Luhmann, N. (1981). Kommunikation mit Zettelkasten. In H. Baier et al. (Eds.), *Offentliche Meinung und sozialer Wandel*.
6. Google DeepMind. (2024). Titans: Learning to Memorize at Test Time. *arXiv preprint*.
7. Anthropic. (2024). Model Context Protocol Specification. https://modelcontextprotocol.io/

---

<p align="center">
  Built by <a href="https://github.com/Haustorium12">Sean Pembroke</a> at 24K Labs.
</p>
