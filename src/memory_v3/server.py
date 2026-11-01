"""
memory-v3 -- MCP Server
FastMCP server exposing the next-generation memory system.
24 tools across CRUD, search, graphs, lifecycle, and system management.
"""

import hashlib
import json
import os
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastmcp import FastMCP

from . import db
from .config import get_config
from .retrieval.filters import RetrievalFilter, add_provenance, build_eligibility_where
from .security import scan_for_credentials
from .lifecycle.consolidation import run_consolidation
from .lifecycle.hallucination import batch_verify, decay_confidence


def _check_embedding_dim(conn, configured_dim: int) -> None:
    """Exit with a clear error if DB vector dim differs from configured provider dim."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_vec'"
    ).fetchone()
    if row is None:
        return  # table doesn't exist yet — will be created at first write
    match = re.search(r"float\[(\d+)\]", row[0])
    if match is None:
        return  # can't determine dim — don't block startup
    db_dim = int(match.group(1))
    if db_dim != configured_dim:
        print(
            f"\n[memory-v3] ERROR: Embedding dimension mismatch!\n"
            f"  DB was created with dim={db_dim}\n"
            f"  Current provider is configured for dim={configured_dim}\n"
            f"\n"
            f"  This usually means you switched embedding providers or models.\n"
            f"  Mixing dimensions will silently corrupt vector search results.\n"
            f"\n"
            f"  To fix: run  memory-v3 reindex --reembed\n"
            f"  WARNING: --reembed will call your embedding API for every memory.\n"
            f"  Depending on your provider and memory count, this may incur API costs.\n"
            f"  Check https://openai.com/pricing (or your provider's pricing) first.\n",
            file=sys.stderr,
        )
        sys.exit(1)


mcp = FastMCP(
    "memory-v3",
    instructions=(
        "memory-v3: Next-generation brain-inspired memory system. "
        "Multi-graph memory (semantic/temporal/causal/entity), 3-stage retrieval, "
        "ACT-R + surprise scoring, constitutional hierarchy, sleep consolidation, "
        "self-linking Zettelkasten, async writes. 12 research-backed upgrades over v2."
    ),
)


def _prewarm_embedding_model():
    try:
        cfg = get_config()
        if cfg.direct_writes_enabled or cfg.indexing_enabled:
            _get_embedder()
    except Exception:
        pass


# Lazy DB connection
_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = db.init_db()
    return _conn


def get_embedder():
    from .providers import get_embedder

    return get_embedder()


def _get_embedder():
    return get_embedder()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _handoff_expiry_iso(cfg) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=cfg.handoff_ttl_days)).isoformat()


def _direct_write_metadata(
    cfg,
    durable: bool = False,
    admin_override: bool = False,
    namespace: str | None = None,
):
    if durable:
        return {
            "owner": cfg.source_policy_owner,
            "namespace": "durable",
            "sensitivity": "normal",
            "record_kind": "derived",
            "expiry_at": None,
        }, None
    if admin_override:
        return {
            "owner": cfg.source_policy_owner,
            "namespace": "admin",
            "sensitivity": "normal",
            "record_kind": "derived",
            "expiry_at": None,
        }, None
    if namespace is not None:
        return None, {
            "error": "Direct writes always use the handoff namespace",
            "code": "unsupported_direct_write_semantics",
        }
    return {
        "owner": cfg.source_policy_owner,
        "namespace": "handoff",
        "sensitivity": "normal",
        "record_kind": "handoff",
        "expiry_at": _handoff_expiry_iso(cfg),
    }, None


def _safe_json(obj) -> str:
    """Serialize to JSON, handling non-serializable types gracefully."""

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, bytes):
            return f"<{len(o)} bytes>"
        if isinstance(o, set):
            return list(o)
        return str(o)

    return json.dumps(obj, indent=2, default=default)


def _parse_tags(raw) -> list[str]:
    """Parse tags from JSON string or return as-is if already a list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _mem_summary(mem: dict) -> dict:
    """Build a concise summary dict from a full memory row."""
    tags = _parse_tags(mem.get("tags"))
    linked = _parse_tags(mem.get("linked_memories"))
    return {
        "id": mem["id"],
        "content": mem["content"][:300],
        "content_type": mem.get("content_type"),
        "tags": tags,
        "confidence": mem.get("confidence"),
        "protected": bool(mem.get("protected")),
        "governance_layer": mem.get("governance_layer"),
        "memory_layer": mem.get("memory_layer"),
        "structure_type": mem.get("structure_type"),
        "surprise_score": mem.get("surprise_score"),
        "verification_status": mem.get("verification_status"),
        "linked_memories": linked,
        "created_at": mem.get("created_at"),
        "updated_at": mem.get("updated_at"),
        "access_count": mem.get("access_count"),
    }


def _direct_writes_disabled(**extra) -> str:
    result = {
        "error": "Direct writes are disabled by MEMORY_V3_DIRECT_WRITES_ENABLED",
    }
    result.update(extra)
    return _safe_json(result)


def _direct_mutation_disabled(tool: str, **extra) -> str:
    result = {
        "error": f"{tool} is disabled; direct memory is add-only expiring handoffs.",
        "code": "direct_mutation_disabled",
        "deprecated": True,
        "mutated": False,
    }
    result.update(extra)
    return _safe_json(result)


def _graph_mutation_disabled(**extra) -> str:
    result = {
        "error": "Graph mutation is disabled; active memory-v3 is graph-free.",
        "code": "graph_lifecycle_disabled",
        "deprecated": True,
    }
    result.update(extra)
    return _safe_json(result)


def _graph_retrieval_unavailable() -> str:
    return _safe_json(
        {
            "error": "Graph retrieval is disabled; active retrieval is lexical/vector only.",
            "code": "graph_retrieval_disabled",
            "deprecated": True,
            "results": [],
            "data": [],
        }
    )


def _handoff_public_error(action: str, **extra) -> str:
    result = {
        "error": f"{action} failed",
        "code": "handoff_operation_failed",
    }
    result.update(extra)
    return _safe_json(result)


def _source_cleanup_error(source_path: str | None, source_id: str | None, exc: Exception) -> dict:
    return {
        "source_path": source_path,
        "source_id": source_id,
        "exception_class": exc.__class__.__name__,
    }


def _cleanup_absent_sources(conn, owner: str, manifest_paths: set[str]) -> dict:
    rows = conn.execute(
        """
        SELECT source_id, source_file, COUNT(*) AS chunk_count
        FROM memories
        WHERE owner = ? AND record_kind = 'derived' AND archived = 0
        GROUP BY source_id, source_file
        ORDER BY source_file, source_id
        """,
        (owner,),
    ).fetchall()
    result = {
        "deleted_sources": 0,
        "deleted_memories": 0,
        "deletion_errors": [],
    }
    for row in rows:
        source_path = row["source_file"]
        if source_path in manifest_paths:
            continue
        try:
            cleanup = db.cleanup_source(
                conn,
                owner=owner,
                source_id=row["source_id"],
                file_path=source_path,
            )
        except Exception as exc:
            result["deletion_errors"].append(
                _source_cleanup_error(source_path, row["source_id"], exc)
            )
            continue
        result["deleted_sources"] += 1
        result["deleted_memories"] += cleanup["deleted_memories"]
    if not result["deletion_errors"]:
        result.pop("deletion_errors")
    return result


def _validate_filter_value(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _public_retrieval_filter(
    cfg,
    *,
    owner: str | None = None,
    namespace: str | None = None,
    namespaces: list[str] | None = None,
    project_slug: str | None = None,
    path_prefix: str | None = None,
    author: str | None = None,
    sensitivity: str | None = None,
    record_kind: str | None = None,
) -> RetrievalFilter | None:
    configured_owner = _validate_filter_value(
        "MEMORY_V3_SOURCE_POLICY_OWNER",
        cfg.source_policy_owner,
    )
    requested_owner = _validate_filter_value("owner", owner)
    if requested_owner is not None and requested_owner != configured_owner:
        return None
    if namespaces is not None and not isinstance(namespaces, (list, tuple)):
        raise ValueError("namespaces must be a list of non-empty strings")
    requested_namespaces = []
    for value in namespaces or []:
        requested_namespaces.append(_validate_filter_value("namespaces", value))
    single_namespace = _validate_filter_value("namespace", namespace)
    if single_namespace is not None:
        requested_namespaces.append(single_namespace)
    return RetrievalFilter(
        owner=configured_owner,
        namespaces=tuple(dict.fromkeys(requested_namespaces)),
        project_slug=_validate_filter_value("project_slug", project_slug),
        path_prefix=_validate_filter_value("path_prefix", path_prefix),
        author=_validate_filter_value("author", author),
        sensitivity=_validate_filter_value("sensitivity", sensitivity),
        record_kind=_validate_filter_value("record_kind", record_kind),
    )


def _validate_limit(limit: int) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer from 1 to 1000")


_PROVENANCE_KEYS = (
    "source_file",
    "path",
    "source_path",
    "source_hash",
    "source_id",
    "chunk_id",
    "owner",
    "namespace",
    "updated_at",
    "source_updated_at",
    "verification_status",
    "chunk_index",
    "chunk_count",
    "source_line",
    "expiry_at",
    "record_kind",
    "sensitivity",
    "author",
    "project_status",
    "project_updated_at",
)


def _public_memory(mem: dict, *, full_content: bool = False) -> dict:
    row = add_provenance(dict(mem))
    result = {
        "id": row["id"],
        "content": row["content"] if full_content else row["content"][:300],
        "content_type": row.get("content_type"),
        "tags": _parse_tags(row.get("tags")),
        "confidence": row.get("confidence"),
        "protected": bool(row.get("protected")),
        "governance_layer": row.get("governance_layer"),
        "memory_layer": row.get("memory_layer"),
        "structure_type": row.get("structure_type"),
        "final_score": row.get("final_score", row.get("hybrid_score", 0)),
    }
    for key in _PROVENANCE_KEYS:
        result[key] = row.get(key)
    if "bm25_rank" in row:
        result["bm25_rank"] = row["bm25_rank"]
    return result


def _filtered_links(conn, mem: dict, retrieval_filter: RetrievalFilter) -> list[dict]:
    raw_links = _parse_tags(mem.get("linked_memories"))
    links = [
        link
        for link in raw_links
        if isinstance(link, dict) and isinstance(link.get("memory_id"), int)
    ]
    if not links:
        return []
    where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
    output = []
    for link in links:
        row = conn.execute(
            f"SELECT m.id, m.content FROM memories m WHERE m.id = ? AND {where_sql}",
            (link["memory_id"], *params),
        ).fetchone()
        if row is None:
            continue
        output.append(
            {
                "memory_id": row["id"],
                "similarity": link.get("similarity", 0.0),
                "content_preview": (row["content"] or "")[:100],
            }
        )
    return output


# ============================================================================
# CRUD (4 tools)
# ============================================================================


@mcp.tool()
def add_memory(
    content: str,
    content_type: str = "fact",
    tags: list[str] | None = None,
    source: str | None = None,
    protected: bool = False,
    author: str = "claude-code",
    confidence: float = 0.95,
    governance_layer: int | None = None,
    structure_type: str | None = None,
    durable: bool = False,
    admin_override: bool = False,
    namespace: str | None = None,
) -> str:
    """Store a memory for the configured owner.
    By default creates a temporary, expiring handoff (30-day TTL).
    Set durable=True to write a canonical durable fact (no expiry, namespace=durable).
    Set admin_override=True for administrative overrides (namespace=admin)."""
    cfg = get_config()
    if not cfg.direct_writes_enabled:
        return _direct_writes_disabled(memory_id=None)

    direct_metadata, metadata_error = _direct_write_metadata(
        cfg, durable, admin_override, namespace
    )
    if metadata_error is not None:
        return _safe_json({**metadata_error, "memory_id": None})

    conn = _get_conn()

    # Credential scan
    cred_matches = scan_for_credentials(content)
    if cred_matches:
        return _safe_json(
            {
                "error": "Credential detected -- memory rejected",
                "matches": cred_matches[:3],
            }
        )

    # Generate embedding
    embed_fn = _get_embedder()
    embedding = embed_fn.embed(content)

    # Novelty check via cosine against top-5 nearest
    novelty = 1.0
    try:
        from .db import _serialize_f32

        vec_rows = conn.execute(
            """
            SELECT id, distance FROM memory_vec
            WHERE embedding MATCH ?
            ORDER BY distance LIMIT 5
            """,
            (_serialize_f32(embedding),),
        ).fetchall()
        if vec_rows:
            closest_dist = (
                vec_rows[0]["distance"] if isinstance(vec_rows[0], dict) else vec_rows[0][1]
            )
            novelty = min(closest_dist, 1.0)
    except Exception:
        pass

    # Auto-classify governance layer
    tag_set = set(tags or [])
    if governance_layer is None:
        from .scoring.hierarchy import classify_governance_layer

        governance_layer = classify_governance_layer(
            content_type,
            protected=protected,
            tags=tag_set,
        )

    # Auto-detect structure type
    if structure_type is None:
        from .retrieval.structure_tags import detect_structure_type

        structure_type = detect_structure_type(content)

    surprise = 0.5

    # Compute importance
    importance = 0.5 + (surprise * 0.2) + (confidence * 0.1)
    if protected:
        importance = max(importance, 0.9)

    memory_id = db.add_memory(
        conn,
        content=content,
        embedding=embedding,
        content_type=content_type,
        source_file=source,
        author=author,
        confidence=confidence,
        protected=protected,
        tags=tags or [],
        importance_score=importance,
        governance_layer=governance_layer,
        structure_type=structure_type,
        surprise_score=surprise,
        source_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        **direct_metadata,
    )

    return _safe_json(
        {
            "memory_id": memory_id,
            "novelty": round(novelty, 4),
            "importance": round(importance, 4),
            "protected": protected,
            "governance_layer": governance_layer,
            "structure_type": structure_type,
            "surprise_score": round(surprise, 4),
        }
    )


@mcp.tool()
def get(
    memory_id: int,
    owner: str | None = None,
    namespace: str | None = None,
    namespaces: list[str] | None = None,
    project_slug: str | None = None,
    path_prefix: str | None = None,
    author: str | None = None,
    sensitivity: str | None = None,
    record_kind: str | None = None,
) -> str:
    """Get a single memory by ID with full content and metadata."""
    cfg = get_config()
    try:
        retrieval_filter = _public_retrieval_filter(
            cfg,
            owner=owner,
            namespace=namespace,
            namespaces=namespaces,
            project_slug=project_slug,
            path_prefix=path_prefix,
            author=author,
            sensitivity=sensitivity,
            record_kind=record_kind,
        )
    except ValueError as exc:
        return _safe_json({"error": str(exc), "code": "invalid_retrieval_filter"})
    if retrieval_filter is None:
        return _safe_json({"error": f"Memory {memory_id} not found or denied"})

    conn = _get_conn()
    where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
    row = conn.execute(
        f"SELECT m.* FROM memories m WHERE m.id = ? AND {where_sql}",
        (memory_id, *params),
    ).fetchone()
    if row is None:
        return _safe_json({"error": f"Memory {memory_id} not found or denied"})

    mem = dict(row)
    result = _public_memory(mem, full_content=True)
    result["importance_score"] = mem.get("importance_score")
    result["activation_score"] = mem.get("activation_score")
    result["decay_rate"] = mem.get("decay_rate")
    result["last_accessed_at"] = mem.get("last_accessed_at")
    result["last_verified_at"] = mem.get("last_verified_at")
    result["linked_memories"] = _filtered_links(conn, mem, retrieval_filter)

    return _safe_json(result)


@mcp.tool()
def update(
    memory_id: int,
    content: str | None = None,
    tags: list[str] | None = None,
    protected: bool | None = None,
    governance_layer: int | None = None,
    structure_type: str | None = None,
) -> str:
    """Update an existing memory's content, tags, or protected status."""
    return _direct_mutation_disabled("update", updated=False, memory_id=memory_id)


@mcp.tool()
def forget(memory_id: int, reason: str = "manual") -> str:
    """Archive a memory to the graveyard. Protected memories cannot be forgotten."""
    return _direct_mutation_disabled(
        "forget",
        forgotten=False,
        memory_id=memory_id,
        reason=reason,
    )


# ============================================================================
# SEARCH (2 tools)
# ============================================================================


@mcp.tool()
def search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
    owner: str | None = None,
    namespace: str | None = None,
    namespaces: list[str] | None = None,
    project_slug: str | None = None,
    path_prefix: str | None = None,
    author: str | None = None,
    sensitivity: str | None = None,
    record_kind: str | None = None,
) -> str:
    """Hybrid search: BM25 keywords + vector similarity + ACT-R scoring.
    Returns ranked results with scores. Use this for any 'what do I know about X' query."""
    cfg = get_config()
    try:
        _validate_limit(limit)
        retrieval_filter = _public_retrieval_filter(
            cfg,
            owner=owner,
            namespace=namespace,
            namespaces=namespaces,
            project_slug=project_slug,
            path_prefix=path_prefix,
            author=author,
            sensitivity=sensitivity,
            record_kind=record_kind,
        )
    except ValueError as exc:
        return _safe_json({"error": str(exc), "code": "invalid_retrieval_filter"})
    if retrieval_filter is None:
        return _safe_json({"structure_hint": "No results found.", "results": []})

    conn = _get_conn()
    embed_fn = _get_embedder()
    query_embedding = embed_fn.embed(query)

    from .retrieval import search as retrieval_search

    results = retrieval_search(
        conn,
        query,
        query_embedding,
        limit=limit,
        content_type=content_type,
        config=cfg,
        retrieval_filter=retrieval_filter,
    )

    # Parse tags for each result
    for r in results:
        r["tags"] = _parse_tags(r.get("tags"))

    # Build structure hint
    if results:
        from .retrieval.structure_tags import get_structure_hint

        structure_types = [r.get("structure_type", "narrative") for r in results]
        dominant_type = max(set(structure_types), key=structure_types.count)
        hint = get_structure_hint(dominant_type, len(results))
    else:
        hint = "No results found."

    output = [_public_memory(r, full_content=True) for r in results[:limit]]

    return _safe_json({"structure_hint": hint, "results": output})


@mcp.tool()
def keyword_search(
    keywords: str,
    limit: int = 10,
    owner: str | None = None,
    namespace: str | None = None,
    namespaces: list[str] | None = None,
    project_slug: str | None = None,
    path_prefix: str | None = None,
    author: str | None = None,
    sensitivity: str | None = None,
    record_kind: str | None = None,
) -> str:
    """Pure BM25 keyword search. Use for exact terms, file names, error codes."""
    cfg = get_config()
    try:
        _validate_limit(limit)
        retrieval_filter = _public_retrieval_filter(
            cfg,
            owner=owner,
            namespace=namespace,
            namespaces=namespaces,
            project_slug=project_slug,
            path_prefix=path_prefix,
            author=author,
            sensitivity=sensitivity,
            record_kind=record_kind,
        )
    except ValueError as exc:
        return _safe_json({"error": str(exc), "code": "invalid_retrieval_filter"})
    if retrieval_filter is None:
        return _safe_json({"count": 0, "results": []})

    conn = _get_conn()
    results = db.keyword_search(
        conn,
        keywords,
        limit=limit,
        retrieval_filter=retrieval_filter,
    )
    output = [_public_memory(r, full_content=True) for r in results]
    return _safe_json({"count": len(output), "results": output})


# ============================================================================
# GRAPH (5 tools)
# ============================================================================


def _get_graph_manager():
    return None


@mcp.tool()
def graph_search(query: str, top_k: int = 10) -> str:
    """Deprecated graph search stub. Returns no graph data."""
    return _graph_retrieval_unavailable()


@mcp.tool()
def graph_stats_tool() -> str:
    """Deprecated graph stats stub. Returns no graph data."""
    return _graph_retrieval_unavailable()


@mcp.tool()
def get_causal_chain(query: str, max_depth: int = 5) -> str:
    """Deprecated causal graph stub. Returns no graph data."""
    return _graph_retrieval_unavailable()


@mcp.tool()
def get_timeline(
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> str:
    """Deprecated temporal graph stub. Returns no graph data."""
    return _graph_retrieval_unavailable()


@mcp.tool()
def link_memories(memory_id_a: int, memory_id_b: int, description: str = "") -> str:
    """Deprecated graph/link mutation stub. Does not mutate DB or graph artifacts."""
    return _graph_mutation_disabled(
        linked=False,
        memory_id_a=memory_id_a,
        memory_id_b=memory_id_b,
    )


# ============================================================================
# SYSTEM (5 tools)
# ============================================================================


@mcp.tool()
def stats() -> str:
    """Get memory system statistics."""
    conn = _get_conn()
    s = db.get_stats(conn)

    # Add write queue depth
    try:
        wq_row = conn.execute(
            "SELECT COUNT(*) FROM write_queue WHERE status = 'pending'"
        ).fetchone()
        s["write_queue_depth"] = wq_row[0] if wq_row else 0
    except Exception:
        s["write_queue_depth"] = 0

    # Add sensory buffer size
    try:
        sb_row = conn.execute("SELECT COUNT(*) FROM sensory_buffer WHERE processed = 0").fetchone()
        s["sensory_buffer_size"] = sb_row[0] if sb_row else 0
    except Exception:
        s["sensory_buffer_size"] = 0

    # Rename for clarity
    s.setdefault("governance_layer_counts", {})
    s.setdefault("verification_counts", {})

    return _safe_json(s)


@mcp.tool()
def list_recent(hours: int = 24, limit: int = 20) -> str:
    """List recently created memories."""
    cfg = get_config()
    try:
        retrieval_filter = _public_retrieval_filter(cfg)
    except ValueError as exc:
        return _safe_json({"error": str(exc), "code": "invalid_retrieval_filter"})
    if retrieval_filter is None:
        return _safe_json({"count": 0, "memories": []})
    conn = _get_conn()
    try:
        results = db.list_recent(
            conn,
            hours=hours,
            limit=limit,
            retrieval_filter=retrieval_filter,
        )
    except ValueError as exc:
        return _safe_json({"error": str(exc), "code": "invalid_retrieval_filter"})
    output = [_public_memory(r) for r in results]
    return _safe_json({"count": len(output), "memories": output})


@mcp.tool()
def list_topics(limit: int = 20) -> str:
    """List owner-safe content-type topics without reading graph artifacts."""
    cfg = get_config()
    try:
        _validate_limit(limit)
        retrieval_filter = _public_retrieval_filter(cfg)
    except ValueError as exc:
        return _safe_json({"error": str(exc), "code": "invalid_retrieval_filter"})

    conn = _get_conn()
    where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
    rows = conn.execute(
        f"SELECT m.content_type, COUNT(*) AS cnt FROM memories m WHERE {where_sql} "
        "GROUP BY m.content_type ORDER BY cnt DESC, m.content_type LIMIT ?",
        (*params, limit),
    ).fetchall()
    topics = [
        {
            "content_type": row["content_type"],
            "count": row["cnt"],
            "source": "content_type",
        }
        for row in rows
    ]

    return _safe_json({"count": len(topics), "topics": topics})


@mcp.tool()
def reindex(force: bool = False) -> str:
    """Index canonical vault documents after editing them. Incremental by default;
    force=True rebuilds sources that are not already current. Search results cite the source."""
    cfg = get_config()

    if not cfg.indexing_enabled:
        return _safe_json(
            {
                "error": "Indexing is disabled by MEMORY_V3_INDEXING_ENABLED",
                "indexed": 0,
                "skipped": 0,
                "errors": 1,
                "force": force,
            }
        )

    conn = _get_conn()
    vault_path = cfg.vault_path

    if not vault_path or not os.path.isdir(vault_path):
        return _safe_json(
            {
                "error": "No vault path configured or directory not found",
                "vault_path": vault_path,
            }
        )

    vault = Path(vault_path)
    from .indexing import reindex_sources
    from .source_policy import build_source_manifest

    manifest = build_source_manifest(vault, config=cfg)
    cleanup = _cleanup_absent_sources(
        conn,
        cfg.source_policy_owner,
        {record.relative_path for record in manifest},
    )

    result = reindex_sources(
        conn,
        vault,
        manifest,
        _get_embedder,
        embed_model=cfg.embed_model,
        embed_provider=cfg.embed_provider,
        embed_dim=cfg.embed_dim,
        force=force,
    )
    public = {
        "indexed": result["indexed"],
        "skipped": result["skipped"],
        "errors": result["errors"],
        "force": result["force"],
    }
    for key in ("deleted_sources", "deleted_memories", "deletion_errors"):
        if cleanup.get(key):
            public[key] = cleanup[key]
    for key in ("status", "source_errors"):
        if key in result and result[key]:
            public[key] = result[key]
    if result["force"] and result.get("already_current"):
        public["already_current"] = result["already_current"]
    return _safe_json(public)


@mcp.tool()
def check_integrity() -> str:
    """Check vault file integrity against saved manifest. Includes staleness detection."""
    cfg = get_config()
    vault_path = cfg.vault_path

    result = {}

    # Vault manifest check
    if vault_path and os.path.isdir(vault_path):
        from .security import check_integrity as vault_check

        result["vault"] = vault_check(vault_path)
    else:
        result["vault"] = {"error": "No vault path configured"}

    # Staleness detection
    conn = _get_conn()
    try:
        from .lifecycle.hallucination import detect_staleness

        stale = detect_staleness(conn)
        result["stale_memories"] = len(stale)
        result["stale_details"] = stale[:20]
    except Exception as e:
        result["stale_memories"] = 0
        result["stale_error"] = str(e)

    # DB integrity
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        result["db_integrity"] = integrity[0] if integrity else "unknown"
    except Exception:
        result["db_integrity"] = "check_failed"

    return _safe_json(result)


# ============================================================================
# ADVANCED (5 tools)
# ============================================================================


@mcp.tool()
def extract_from_conversation(
    text: str,
    source: str | None = None,
    author: str = "claude-code",
    durable: bool = False,
    admin_override: bool = False,
    namespace: str | None = None,
) -> str:
    """Extract temporary, expiring handoffs from a conversation.
    This does not persist durable facts; edit the canonical vault document and
    run reindex instead. durable=True and admin_override are unsupported."""
    cfg = get_config()
    if not cfg.direct_writes_enabled:
        return _direct_writes_disabled(facts_extracted=0)

    direct_metadata, metadata_error = _direct_write_metadata(
        cfg, durable, admin_override, namespace
    )
    if metadata_error is not None:
        return _safe_json({**metadata_error, "facts_extracted": 0})

    conn = _get_conn()

    # Route through sensory filter first if enabled
    if cfg.enable_sensory:
        from .lifecycle.sensory import SensoryFilter

        sf = SensoryFilter(conn, config=cfg)
        gate = sf.process(content=text, source=source, author=author)
        if not gate["accepted"]:
            return _safe_json(
                {
                    "sensory_filtered": True,
                    "reason": gate["reason"],
                    "facts_extracted": 0,
                }
            )

    from .lifecycle.extraction import process_conversation

    result = process_conversation(
        conn,
        text,
        source=source,
        author=author,
        config=cfg,
        add_only=True,
        memory_metadata=direct_metadata,
    )
    return _safe_json(result)


@mcp.tool()
def compact_text(text: str, max_ratio: float = 3.0, verify: bool = True) -> str:
    """Compact text using CogCanvas pipeline: extract protected content,
    delete expendable, summarize remainder, verify faithfulness."""
    return _direct_mutation_disabled(
        "compact_text",
        compressed=None,
        protected_items=[],
    )


@mcp.tool()
def list_active_handoffs(
    agent_id: str | None = None,
    since: str | None = None,
    now: str | None = None,
) -> str:
    """List unexpired handoffs for the configured owner."""
    try:
        rows = db.list_active_handoffs(_get_conn(), agent_id=agent_id, since=since, now=now)
    except Exception:
        return _handoff_public_error("List active handoffs", count=0, handoffs=[])
    return _safe_json({"count": len(rows), "handoffs": rows})


@mcp.tool()
def sweep_expired_handoffs(now: str | None = None) -> str:
    """Delete expired handoffs when the independent sweeper gate is enabled."""
    cfg = get_config()
    if not cfg.handoff_sweeper_enabled:
        return _safe_json({"enabled": False, "deleted": 0})
    try:
        deleted = db.sweep_expired_handoffs(_get_conn(), now=now)
    except Exception:
        return _handoff_public_error("Sweep expired handoffs", enabled=True, deleted=0)
    return _safe_json({"enabled": True, "deleted": deleted})


@mcp.tool()
def rebuild_graph_debt() -> str:
    """Deprecated graph debt rebuild stub. Does not access DB or graph artifacts."""
    return _graph_mutation_disabled(status="disabled", debt_resolved=0, data=[])


@mcp.tool()
def agent_sync(agent_id: str) -> str:
    """Deprecated sync API returning active handoffs by default."""
    try:
        conn = _get_conn()
        from .multi_agent import sync_agent

        return _safe_json(sync_agent(agent_id, conn))
    except Exception:
        return _safe_json(
            {
                "error": "Agent synchronization failed",
                "deprecated": True,
                "mode": "active_handoffs",
                "compatibility_mode": False,
            }
        )


@mcp.tool()
def check_conflicts() -> str:
    """Get unresolved memory conflicts between agents."""
    conn = _get_conn()
    from .multi_agent import get_unresolved_conflicts

    conflicts = get_unresolved_conflicts(conn)
    return _safe_json({"count": len(conflicts), "conflicts": conflicts})


@mcp.tool()
def decay_sweep() -> str:
    """Run FadeMem decay sweep: update importance scores,
    promote/demote between STM/LTM, archive low-importance memories."""
    cfg = get_config()
    if not cfg.direct_writes_enabled:
        return _direct_writes_disabled(
            importance_updated=0,
            layers_changed=0,
            archived=0,
            confidence_decayed=0,
        )

    conn = _get_conn()

    # Phase 1: consolidation (replay, reorganize, compress, prune)
    consolidation = run_consolidation(conn, cycle_type="full", config=cfg)

    # Phase 2: HaluMem verification + confidence decay
    verification = batch_verify(conn, max_check=50)
    confidence_decay = decay_confidence(conn, config=cfg)

    # Phase 3: sweep expired handoffs
    swept = db.sweep_expired_handoffs(conn)

    return _safe_json(
        {
            "status": "ok",
            "deprecated": False,
            "importance_updated": consolidation.get("importance_updated", 0),
            "layers_changed": consolidation.get("layers_changed", 0),
            "archived": consolidation.get("archived", 0) + swept,
            "confidence_decayed": confidence_decay.get("memories_decayed", 0),
            "verified": verification.get("verified", 0),
            "stale": verification.get("stale", 0),
            "unverified": verification.get("unverified", 0),
            "handoffs_swept": swept,
            "consolidation": consolidation,
            "verification": verification,
        }
    )


# ============================================================================
# NEW TOOLS (3)
# ============================================================================


@mcp.tool()
def sleep_consolidation(cycle_type: str = "full") -> str:
    """Run consolidation cycle: replay, reorganize, compress, prune, or full.
    Returns stats per phase."""
    cfg = get_config()
    if not cfg.direct_writes_enabled:
        return _direct_writes_disabled(cycle_type=cycle_type, stats={})

    conn = _get_conn()
    result = run_consolidation(conn, cycle_type=cycle_type, config=cfg)
    return _safe_json(result)


@mcp.tool()
def verify_memories(memory_ids: list[int] | None = None, max_check: int = 50) -> str:
    """HaluMem: check source files, update confidence and verification_status."""
    cfg = get_config()
    if not cfg.direct_writes_enabled:
        return _direct_writes_disabled(
            checked=0,
            verified=0,
            stale=0,
            unverified=0,
            errors=0,
            details=[],
        )

    conn = _get_conn()
    result = batch_verify(conn, memory_ids=memory_ids, max_check=max_check)
    return _safe_json(result)


@mcp.tool()
def set_governance_layer(memory_id: int, layer: int) -> str:
    """Change governance tier (1-4). Layer 1 requires human authority level."""
    cfg = get_config()
    if not cfg.direct_writes_enabled:
        return _direct_writes_disabled(
            memory_id=memory_id,
            old_layer=None,
            new_layer=layer,
            auto_protected=False,
        )

    if layer < 1 or layer > 4:
        return _safe_json(
            {
                "error": "Invalid governance layer",
                "code": "invalid_layer",
                "memory_id": memory_id,
            }
        )

    conn = _get_conn()
    row = conn.execute(
        "SELECT governance_layer FROM memories WHERE id = ? AND archived = 0",
        (memory_id,),
    ).fetchone()
    if row is None:
        return _safe_json(
            {
                "error": "Memory not found",
                "code": "not_found",
                "memory_id": memory_id,
            }
        )

    old_layer = row[0] if not isinstance(row, dict) else row["governance_layer"]
    update_memory(conn, memory_id, governance_layer=layer)
    return _safe_json(
        {
            "memory_id": memory_id,
            "old_layer": old_layer,
            "new_layer": layer,
            "auto_protected": layer == 1,
        }
    )


# ============================================================================
# Entry point
# ============================================================================


def run():
    """Entry point for the MCP server."""
    cfg = get_config()
    conn = _get_conn()
    _check_embedding_dim(conn, cfg.embed_dim)
    threading.Thread(target=_prewarm_embedding_model, daemon=True).start()
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run()
