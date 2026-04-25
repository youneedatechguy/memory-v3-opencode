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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastmcp import FastMCP

from . import db, embeddings, get_embedder
from .config import get_config, GOVERNANCE_LAYERS
from .security import scan_for_credentials


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

# Pre-warm embedding model in background
def _prewarm():
    try:
        get_embedder()
    except Exception:
        pass

threading.Thread(target=_prewarm, daemon=True).start()

# Lazy DB connection
_conn = None

def _get_conn():
    global _conn
    if _conn is None:
        _conn = db.init_db()
    return _conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


# ============================================================================
# CRUD (4 tools)
# ============================================================================

@mcp.tool()
def add_memory(
    content: str,
    content_type: str = "fact",
    tags: Optional[list[str]] = None,
    source: Optional[str] = None,
    protected: bool = False,
    author: str = "claude-code",
    confidence: float = 0.95,
    governance_layer: Optional[int] = None,
    structure_type: Optional[str] = None,
) -> str:
    """Store a new memory. Types: fact, episode, decision, correction, identity, person.
    Protected memories never decay. Returns the memory ID."""
    conn = _get_conn()
    cfg = get_config()

    # Credential scan
    cred_matches = scan_for_credentials(content)
    if cred_matches:
        return _safe_json({
            "error": "Credential detected -- memory rejected",
            "matches": cred_matches[:3],
        })

    # Generate embedding
    embed_fn = get_embedder()
    embedding = embed_fn.embed(content)

    # Novelty check via cosine against top-5 nearest
    novelty = 1.0
    try:
        from .db import _serialize_f32
        vec_rows = conn.execute(
            "SELECT id, distance FROM memory_vec WHERE embedding MATCH ? ORDER BY distance LIMIT 5",
            (_serialize_f32(embedding),),
        ).fetchall()
        if vec_rows:
            closest_dist = vec_rows[0]["distance"] if isinstance(vec_rows[0], dict) else vec_rows[0][1]
            novelty = min(closest_dist, 1.0)
    except Exception:
        pass

    # Auto-classify governance layer
    tag_set = set(tags or [])
    if governance_layer is None:
        from .scoring.hierarchy import classify_governance_layer
        governance_layer = classify_governance_layer(content_type, protected=protected, tags=tag_set)

    # Auto-detect structure type
    if structure_type is None:
        from .retrieval.structure_tags import detect_structure_type
        structure_type = detect_structure_type(content)

    # Compute surprise score
    surprise = 0.5
    if cfg.enable_surprise:
        try:
            from .scoring.surprise import compute_surprise, get_centroids
            centroid_pairs = get_centroids(conn)
            centroids = [c for _, c in centroid_pairs]
            surprise = compute_surprise(np.array(embedding, dtype=np.float32), centroids)
        except Exception:
            pass

    # Compute importance
    importance = 0.5 + (surprise * 0.2) + (confidence * 0.1)
    if protected:
        importance = max(importance, 0.9)

    # Async vs sync write
    if cfg.enable_async and cfg.async_enabled:
        try:
            from .async_ops.write_queue import enqueue
            payload = {
                "content": content,
                "embedding": embedding,
                "kwargs": {
                    "content_type": content_type,
                    "source_file": source,
                    "author": author,
                    "confidence": confidence,
                    "protected": protected,
                    "tags": tags or [],
                    "importance_score": importance,
                    "governance_layer": governance_layer,
                    "structure_type": structure_type,
                    "surprise_score": surprise,
                    "source_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
                },
            }
            priority = governance_layer or 3
            write_id = enqueue(conn, "add_memory", payload, priority=priority)
            # Flush immediately for sync-like behavior
            from .async_ops.write_queue import flush
            flush(conn, batch_size=1)
            # Get the memory ID from the last insert
            row = conn.execute("SELECT MAX(id) FROM memories").fetchone()
            memory_id = row[0] if row else write_id
        except Exception:
            # Fallback to sync
            memory_id = db.add_memory(
                conn, content=content, embedding=embedding,
                content_type=content_type, source_file=source, author=author,
                confidence=confidence, protected=protected, tags=tags or [],
                importance_score=importance, governance_layer=governance_layer,
                structure_type=structure_type, surprise_score=surprise,
                source_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            )
    else:
        memory_id = db.add_memory(
            conn, content=content, embedding=embedding,
            content_type=content_type, source_file=source, author=author,
            confidence=confidence, protected=protected, tags=tags or [],
            importance_score=importance, governance_layer=governance_layer,
            structure_type=structure_type, surprise_score=surprise,
            source_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        )

    # Zettelkasten auto-link in background
    if cfg.enable_zettelkasten:
        def _auto_link():
            try:
                from .graphs.zettelkasten import auto_link
                auto_link(conn, memory_id, embedding, threshold=cfg.zettel_link_threshold)
            except Exception:
                pass
        threading.Thread(target=_auto_link, daemon=True).start()

    return _safe_json({
        "memory_id": memory_id,
        "novelty": round(novelty, 4),
        "importance": round(importance, 4),
        "protected": protected,
        "governance_layer": governance_layer,
        "structure_type": structure_type,
        "surprise_score": round(surprise, 4),
    })


@mcp.tool()
def get(memory_id: int) -> str:
    """Get a single memory by ID with full content and metadata."""
    conn = _get_conn()
    mem = db.get_memory(conn, memory_id)
    if mem is None:
        return _safe_json({"error": f"Memory {memory_id} not found or archived"})

    result = _mem_summary(mem)
    result["content"] = mem["content"]  # full content
    result["source_file"] = mem.get("source_file")
    result["author"] = mem.get("author")
    result["importance_score"] = mem.get("importance_score")
    result["activation_score"] = mem.get("activation_score")
    result["decay_rate"] = mem.get("decay_rate")
    result["last_accessed_at"] = mem.get("last_accessed_at")
    result["last_verified_at"] = mem.get("last_verified_at")

    # Get linked memory details
    try:
        from .graphs.zettelkasten import get_links
        result["linked_memories"] = get_links(conn, memory_id)
    except Exception:
        result["linked_memories"] = _parse_tags(mem.get("linked_memories"))

    return _safe_json(result)


@mcp.tool()
def update(
    memory_id: int,
    content: Optional[str] = None,
    tags: Optional[list[str]] = None,
    protected: Optional[bool] = None,
    governance_layer: Optional[int] = None,
    structure_type: Optional[str] = None,
) -> str:
    """Update an existing memory's content, tags, or protected status."""
    conn = _get_conn()

    # Validate governance layer
    if governance_layer is not None and governance_layer not in (1, 2, 3, 4):
        return _safe_json({"error": "governance_layer must be 1-4"})

    success = db.update_memory(
        conn, memory_id,
        content=content, tags=tags, protected=protected,
        governance_layer=governance_layer, structure_type=structure_type,
    )

    if not success:
        return _safe_json({"error": f"Memory {memory_id} not found or no fields to update"})

    # Re-embed if content changed
    if content is not None:
        try:
            embed_fn = get_embedder()
            embedding = embed_fn.embed(content)
            from .db import _serialize_f32
            conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                (memory_id, _serialize_f32(embedding)),
            )
            conn.commit()
        except Exception:
            pass

    mem = db.get_memory(conn, memory_id)
    return _safe_json({"updated": True, "memory": _mem_summary(mem) if mem else None})


@mcp.tool()
def forget(memory_id: int, reason: str = "manual") -> str:
    """Archive a memory to the graveyard. Protected memories cannot be forgotten."""
    conn = _get_conn()

    # Check if memory exists and get its governance layer
    row = conn.execute(
        "SELECT governance_layer, protected, content FROM memories WHERE id = ? AND archived = 0",
        (memory_id,),
    ).fetchone()
    if row is None:
        return _safe_json({"error": f"Memory {memory_id} not found or already archived"})

    mem = dict(row)
    gov_layer = mem.get("governance_layer", 3)

    # Layer 1 (constitutional) cannot be forgotten
    if gov_layer == 1:
        return _safe_json({
            "error": "Cannot forget a Layer 1 (Constitutional) memory. "
                     "Constitutional memories are permanent by design -- they encode "
                     "identity, chain of command, and core rules. To remove this memory, "
                     "first change its governance layer with set_governance_layer.",
            "memory_id": memory_id,
            "governance_layer": 1,
        })

    if mem.get("protected"):
        return _safe_json({
            "error": "Cannot forget a protected memory. Unprotect it first via update().",
            "memory_id": memory_id,
        })

    success = db.archive_memory(conn, memory_id, reason=reason)
    return _safe_json({"forgotten": success, "memory_id": memory_id, "reason": reason})


# ============================================================================
# SEARCH (2 tools)
# ============================================================================

@mcp.tool()
def search(query: str, limit: int = 10, content_type: Optional[str] = None) -> str:
    """Hybrid search: BM25 keywords + vector similarity + ACT-R scoring.
    Returns ranked results with scores. Use this for any 'what do I know about X' query."""
    conn = _get_conn()
    cfg = get_config()

    embed_fn = get_embedder()
    query_embedding = embed_fn.embed(query)

    if cfg.enable_three_stage:
        from .retrieval import search as retrieval_search
        results = retrieval_search(
            conn, query, query_embedding, limit=limit,
            content_type=content_type, config=cfg,
        )
    else:
        results = db.hybrid_search(
            conn, query, query_embedding, limit=limit, content_type=content_type,
        )
        from .scoring import rank_results
        results = rank_results(conn, results, query)

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

    output = []
    for r in results[:limit]:
        entry = {
            "id": r["id"],
            "content": r["content"][:300],
            "content_type": r.get("content_type"),
            "tags": r.get("tags"),
            "confidence": r.get("confidence"),
            "governance_layer": r.get("governance_layer"),
            "structure_type": r.get("structure_type"),
            "final_score": r.get("final_score", r.get("hybrid_score", 0)),
            "see_also": [
                link.get("memory_id")
                for link in _parse_tags(r.get("linked_memories"))
                if isinstance(link, dict)
            ][:3],
        }
        # Include graph-related fields if present
        if "related_nodes" in r:
            entry["related_nodes"] = r["related_nodes"]
        output.append(entry)

    return _safe_json({"structure_hint": hint, "results": output})


@mcp.tool()
def keyword_search(keywords: str, limit: int = 10) -> str:
    """Pure BM25 keyword search. Use for exact terms, file names, error codes."""
    conn = _get_conn()
    results = db.keyword_search(conn, keywords, limit=limit)
    output = []
    for r in results:
        output.append({
            "id": r["id"],
            "content": r["content"][:300],
            "content_type": r.get("content_type"),
            "tags": _parse_tags(r.get("tags")),
            "bm25_rank": r.get("bm25_rank"),
            "governance_layer": r.get("governance_layer"),
        })
    return _safe_json({"count": len(output), "results": output})


# ============================================================================
# GRAPH (5 tools)
# ============================================================================

def _get_graph_manager():
    """Lazy-load the GraphManager singleton."""
    from .graphs import GraphManager
    cfg = get_config()
    return GraphManager(graph_dir=cfg.graph_dir)


@mcp.tool()
def graph_search(query: str, top_k: int = 10) -> str:
    """Knowledge graph search using Personalized PageRank.
    Discovers related concepts via graph traversal (multi-hop reasoning)."""
    gm = _get_graph_manager()
    results = gm.query(query, top_k=top_k, embeddings_module=embeddings)
    output = []
    for r in results:
        output.append({
            "node": r.get("node"),
            "score": r.get("score"),
            "node_type": r.get("node_type"),
            "graph_layer": r.get("layer", "semantic"),
            "source_file": r.get("source_file"),
        })
    return _safe_json({"count": len(output), "results": output})


@mcp.tool()
def graph_stats_tool() -> str:
    """Get knowledge graph statistics: node/edge counts, types, top entities."""
    gm = _get_graph_manager()
    stats = gm.stats()
    return _safe_json(stats)


@mcp.tool()
def get_causal_chain(query: str, max_depth: int = 5) -> str:
    """Search causal graph for query-matched nodes, then trace root-cause-to-effect chain."""
    gm = _get_graph_manager()

    # First search causal graph for matching nodes
    causal = gm._get_layer("causal")
    search_results = causal.search(query, top_k=3, embeddings_module=embeddings)

    if not search_results:
        return _safe_json({"chain": [], "message": "No matching nodes found in causal graph"})

    # Get the causal chain from the best match
    start_node = search_results[0]["node"]
    chain = causal.get_causal_chain(start_node, max_depth=max_depth)

    output = [{
        "node": start_node,
        "depth": 0,
        "role": "start",
        "node_type": search_results[0].get("node_type"),
    }]
    for item in chain:
        output.append({
            "node": item["node"],
            "depth": item["depth"],
            "edge_type": item.get("edge_type"),
            "direction": item.get("direction"),
            "node_type": item.get("node_type"),
        })

    return _safe_json({"start_node": start_node, "chain": output})


@mcp.tool()
def get_timeline(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Query temporal graph for chronologically ordered events. Optional date range filter."""
    gm = _get_graph_manager()
    temporal = gm._get_layer("temporal")

    # Search for query-relevant events first
    search_results = temporal.search(query, top_k=limit * 2, embeddings_module=embeddings)

    # Also get timeline with date filter
    timeline = temporal.get_timeline(start_date=start_date, end_date=end_date, limit=limit)

    # Merge: prioritize search-matched events, then fill with timeline
    seen = set()
    output = []

    for r in search_results:
        node = r["node"]
        if node not in seen:
            ts = r.get("timestamp", r.get("created_at", ""))
            # Apply date filter
            if start_date and ts < start_date:
                continue
            if end_date and ts > end_date:
                continue
            seen.add(node)
            output.append({
                "node": node,
                "timestamp": ts,
                "node_type": r.get("node_type"),
                "source_file": r.get("source_file"),
                "ppr_score": r.get("score"),
            })

    for event in timeline:
        node = event["node"]
        if node not in seen:
            seen.add(node)
            output.append({
                "node": node,
                "timestamp": event.get("timestamp", ""),
                "node_type": event.get("node_type"),
                "source_file": event.get("source_file"),
            })

    # Sort chronologically
    output.sort(key=lambda e: e.get("timestamp", ""))
    return _safe_json({"count": len(output[:limit]), "events": output[:limit]})


@mcp.tool()
def link_memories(memory_id_a: int, memory_id_b: int, description: str = "") -> str:
    """Create manual bidirectional zettelkasten link between two memories."""
    conn = _get_conn()

    from .graphs.zettelkasten import add_manual_link
    success = add_manual_link(conn, memory_id_a, memory_id_b, description=description)

    if not success:
        return _safe_json({
            "error": "Failed to link -- one or both memories not found",
            "memory_id_a": memory_id_a,
            "memory_id_b": memory_id_b,
        })

    # Also add edge to semantic graph
    try:
        gm = _get_graph_manager()
        from .graphs.base import add_edge_to_graph
        G = gm.get_graph("semantic")
        add_edge_to_graph(
            G, f"memory:{memory_id_a}", f"memory:{memory_id_b}",
            "similar_to", description=description or "Manual link",
            confidence=1.0, timestamp=_now_iso(),
        )
        gm.save_all()
    except Exception:
        pass

    return _safe_json({
        "linked": True,
        "memory_id_a": memory_id_a,
        "memory_id_b": memory_id_b,
        "description": description or "Manual link",
    })


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
        sb_row = conn.execute(
            "SELECT COUNT(*) FROM sensory_buffer WHERE processed = 0"
        ).fetchone()
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
    conn = _get_conn()
    results = db.list_recent(conn, hours=hours, limit=limit)
    output = []
    for r in results:
        output.append({
            "id": r["id"],
            "content": r["content"][:200],
            "content_type": r.get("content_type"),
            "tags": _parse_tags(r.get("tags")),
            "governance_layer": r.get("governance_layer"),
            "created_at": r.get("created_at"),
        })
    return _safe_json({"count": len(output), "memories": output})


@mcp.tool()
def list_topics(limit: int = 20) -> str:
    """List topic clusters from the knowledge graph (if built) or content types."""
    conn = _get_conn()

    topics = []

    # Try graph communities first
    try:
        gm = _get_graph_manager()
        semantic = gm._get_layer("semantic")
        G = semantic.G
        if G.number_of_nodes() > 0:
            import networkx as nx
            # Use connected components as rough topics
            undirected = G.to_undirected()
            for i, component in enumerate(
                sorted(nx.connected_components(undirected), key=len, reverse=True)
            ):
                if i >= limit:
                    break
                nodes = sorted(component, key=lambda n: G.degree(n), reverse=True)
                topics.append({
                    "cluster_id": i,
                    "size": len(nodes),
                    "top_nodes": nodes[:5],
                    "source": "graph",
                })
    except Exception:
        pass

    # Fallback: content type distribution
    if not topics:
        rows = conn.execute(
            "SELECT content_type, COUNT(*) as cnt FROM memories "
            "WHERE archived = 0 GROUP BY content_type ORDER BY cnt DESC"
        ).fetchall()
        for row in rows[:limit]:
            topics.append({
                "content_type": row["content_type"],
                "count": row["cnt"],
                "source": "content_type",
            })

    return _safe_json({"count": len(topics), "topics": topics})


@mcp.tool()
def reindex(force: bool = False) -> str:
    """Re-index the vault. Incremental by default (only changed files).
    Set force=True to re-index everything."""
    conn = _get_conn()
    cfg = get_config()
    vault_path = cfg.vault_path

    if not vault_path or not os.path.isdir(vault_path):
        return _safe_json({
            "error": "No vault path configured or directory not found",
            "vault_path": vault_path,
        })

    from .lifecycle.extraction import process_conversation, _extract_tags, detect_content_type
    from .scoring.hierarchy import classify_governance_layer

    indexed = 0
    skipped = 0
    errors = 0

    vault = Path(vault_path)
    embed_fn = get_embedder()

    for md_file in vault.rglob("*.md"):
        rel_path = str(md_file.relative_to(vault)).replace("\\", "/")

        # Skip changelog and manifest
        if rel_path in ("changelog.md", "manifest.json"):
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            errors += 1
            continue

        # Check if already indexed with same hash
        if not force:
            row = conn.execute(
                "SELECT content_hash FROM file_hashes WHERE file_path = ?",
                (rel_path,),
            ).fetchone()
            if row and (row["content_hash"] if isinstance(row, dict) else row[0]) == content_hash:
                skipped += 1
                continue

            # Safety net: if any memory already exists for this source_file,
            # skip it. Prevents duplicate ingestion when file_hashes table is
            # missing entries (e.g. post-migration).
            existing = conn.execute(
                "SELECT 1 FROM memories WHERE source_file = ? LIMIT 1",
                (rel_path,),
            ).fetchone()
            if existing:
                # Backfill the file_hashes entry so future runs skip fast
                conn.execute(
                    """INSERT INTO file_hashes (file_path, content_hash, indexed_at, chunk_count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(file_path) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        indexed_at = excluded.indexed_at""",
                    (rel_path, content_hash, _now_iso()),
                )
                conn.commit()
                skipped += 1
                continue

        # Index the file as a single memory
        try:
            embedding = embed_fn.embed(content[:2000])  # Cap embedding input
            ct = detect_content_type(content)
            tags = _extract_tags(content)
            gov = classify_governance_layer(ct, tags=set(tags))

            db.add_memory(
                conn, content=content[:5000], embedding=embedding,
                content_type=ct, source_file=rel_path, author="indexer",
                tags=tags, governance_layer=gov,
                source_hash=content_hash,
            )

            # Update file hash
            conn.execute(
                """INSERT INTO file_hashes (file_path, content_hash, indexed_at, chunk_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(file_path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    indexed_at = excluded.indexed_at""",
                (rel_path, content_hash, _now_iso()),
            )
            conn.commit()
            indexed += 1
        except Exception as e:
            errors += 1

    return _safe_json({
        "indexed": indexed,
        "skipped": skipped,
        "errors": errors,
        "force": force,
    })


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
    source: Optional[str] = None,
    author: str = "claude-code",
) -> str:
    """Auto-extract facts from a conversation and store them.
    Runs the 2-pass pipeline: extract facts, then decide ADD/UPDATE/DELETE/NONE."""
    conn = _get_conn()
    cfg = get_config()

    # Route through sensory filter first if enabled
    if cfg.enable_sensory:
        from .lifecycle.sensory import SensoryFilter
        sf = SensoryFilter(conn, config=cfg)
        gate = sf.process(content=text, source=source, author=author)
        if not gate["accepted"]:
            return _safe_json({
                "sensory_filtered": True,
                "reason": gate["reason"],
                "facts_extracted": 0,
            })

    from .lifecycle.extraction import process_conversation
    result = process_conversation(conn, text, source=source, author=author, config=cfg)
    return _safe_json(result)


@mcp.tool()
def compact_text(text: str, max_ratio: float = 3.0, verify: bool = True) -> str:
    """Compact text using CogCanvas pipeline: extract protected content,
    delete expendable, summarize remainder, verify faithfulness."""
    conn = _get_conn()
    cfg = get_config()

    from .lifecycle.compaction import compact
    result = compact(text, max_ratio=max_ratio, verify=verify, conn=conn, config=cfg)
    return _safe_json({
        "compressed": result["compressed"],
        "protected_items": result["protected_items"],
        "verification": result["verification"],
        "receipt": {
            "original_length": result["receipt"]["original_length"],
            "compressed_length": result["receipt"]["compressed_length"],
            "ratio": result["receipt"]["ratio"],
        },
    })


@mcp.tool()
def agent_sync(agent_id: str) -> str:
    """Sync an agent: get unread changelog entries and update offset.
    Use on boot to catch up on what other agents did."""
    conn = _get_conn()
    from .multi_agent import sync_agent
    result = sync_agent(agent_id, conn)
    return _safe_json(result)


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
    conn = _get_conn()
    cfg = get_config()

    sweep_stats = {"importance_updated": 0, "layers_changed": 0, "archived": 0, "confidence_decayed": 0}

    # Get max access count for normalization
    max_ac_row = conn.execute(
        "SELECT MAX(access_count) FROM memories WHERE archived = 0"
    ).fetchone()
    max_access_count = (max_ac_row[0] or 1) if max_ac_row else 1

    now = datetime.now(timezone.utc)

    from .scoring.fademem import importance_score as calc_importance, should_archive, get_memory_layer

    # Process all active memories
    rows = conn.execute(
        """SELECT id, content, importance_score, access_count,
                  last_accessed_at, created_at, tags, governance_layer,
                  surprise_score, decay_rate, memory_layer, protected
        FROM memories WHERE archived = 0"""
    ).fetchall()

    for row in rows:
        mem = dict(row)
        mid = mem["id"]

        if mem.get("protected") or mem.get("governance_layer", 3) in (1, 2):
            continue

        tags = set(_parse_tags(mem.get("tags")))

        try:
            created = datetime.fromisoformat(mem["created_at"].replace("Z", "+00:00"))
            age_days = (now - created).total_seconds() / 86400
        except (ValueError, TypeError):
            age_days = 0

        try:
            last_access = datetime.fromisoformat(
                mem["last_accessed_at"].replace("Z", "+00:00")
            )
            hours_since = (now - last_access).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_since = age_days * 24

        # Layer-aware decay rate
        gov_layer = mem.get("governance_layer", 3)
        layer_cfg = GOVERNANCE_LAYERS.get(gov_layer)
        decay_rate = layer_cfg.decay_rate if layer_cfg else mem.get("decay_rate", 0.1)

        # Recalculate importance with surprise
        new_imp = calc_importance(
            relevance=0.4,
            access_count=mem.get("access_count", 1),
            max_access_count=max_access_count,
            hours_since_access=hours_since,
            surprise_score=mem.get("surprise_score", 0.5),
            decay_rate=decay_rate,
        )

        # Update importance if changed significantly
        old_imp = mem.get("importance_score", 0.5)
        if abs(new_imp - old_imp) > 0.01:
            conn.execute(
                "UPDATE memories SET importance_score = ?, updated_at = ? WHERE id = ?",
                (new_imp, _now_iso(), mid),
            )
            sweep_stats["importance_updated"] += 1

        # Update memory layer (STM/LTM)
        new_layer = get_memory_layer(new_imp)
        old_layer = mem.get("memory_layer", "STM")
        if new_layer != old_layer and new_layer != "current":
            conn.execute(
                "UPDATE memories SET memory_layer = ?, updated_at = ? WHERE id = ?",
                (new_layer, _now_iso(), mid),
            )
            sweep_stats["layers_changed"] += 1

        # Check for archival
        if should_archive(new_imp, age_days, tags=tags, governance_layer=gov_layer):
            db.archive_memory(conn, mid, reason="decay_sweep")
            sweep_stats["archived"] += 1

    conn.commit()

    # Confidence decay for stale memories
    if cfg.enable_halumem:
        try:
            from .lifecycle.hallucination import decay_confidence
            conf_result = decay_confidence(conn, config=cfg)
            sweep_stats["confidence_decayed"] = conf_result.get("memories_decayed", 0)
        except Exception:
            pass

    return _safe_json(sweep_stats)


# ============================================================================
# NEW TOOLS (3)
# ============================================================================

@mcp.tool()
def sleep_consolidation(cycle_type: str = "full") -> str:
    """Run consolidation cycle: replay, reorganize, compress, prune, or full.
    Returns stats per phase."""
    conn = _get_conn()
    cfg = get_config()

    valid_types = ("replay", "reorganize", "compress", "prune", "full")
    if cycle_type not in valid_types:
        return _safe_json({"error": f"Invalid cycle_type. Must be one of: {valid_types}"})

    from .lifecycle.consolidation import run_consolidation
    result = run_consolidation(conn, cycle_type=cycle_type, config=cfg)

    # Save graphs after consolidation
    try:
        gm = _get_graph_manager()
        gm.save_all()
    except Exception:
        pass

    return _safe_json({"cycle_type": cycle_type, "stats": result})


@mcp.tool()
def verify_memories(memory_ids: Optional[list[int]] = None, max_check: int = 50) -> str:
    """HaluMem: check source files, update confidence and verification_status."""
    conn = _get_conn()

    from .lifecycle.hallucination import batch_verify
    result = batch_verify(conn, memory_ids=memory_ids, max_check=max_check)

    # Summarize without full details (can be large)
    return _safe_json({
        "checked": result["checked"],
        "verified": result["verified"],
        "stale": result["stale"],
        "unverified": result["unverified"],
        "errors": result["errors"],
        "details": result["details"][:20],
    })


@mcp.tool()
def set_governance_layer(memory_id: int, layer: int) -> str:
    """Change governance tier (1-4). Layer 1 requires human authority level."""
    conn = _get_conn()

    if layer not in (1, 2, 3, 4):
        return _safe_json({"error": "layer must be 1-4"})

    # Check memory exists
    row = conn.execute(
        "SELECT id, governance_layer, content, protected FROM memories WHERE id = ? AND archived = 0",
        (memory_id,),
    ).fetchone()
    if row is None:
        return _safe_json({"error": f"Memory {memory_id} not found or archived"})

    mem = dict(row)
    old_layer = mem.get("governance_layer", 3)

    # Layer 1 promotion requires noting this is a significant action
    if layer == 1 and old_layer != 1:
        # Auto-protect when promoting to constitutional
        db.update_memory(conn, memory_id, governance_layer=layer, protected=True)
    elif layer > 2 and old_layer <= 2:
        # Demoting from constitutional/legislative -- allowed but noteworthy
        db.update_memory(conn, memory_id, governance_layer=layer)
    else:
        db.update_memory(conn, memory_id, governance_layer=layer)

    layer_names = {1: "Constitutional", 2: "Legislative", 3: "Factual", 4: "Ephemeral"}
    return _safe_json({
        "memory_id": memory_id,
        "old_layer": old_layer,
        "new_layer": layer,
        "old_layer_name": layer_names.get(old_layer, "Unknown"),
        "new_layer_name": layer_names.get(layer, "Unknown"),
        "auto_protected": layer == 1,
    })


# ============================================================================
# Entry point
# ============================================================================

def run():
    """Entry point for the MCP server."""
    cfg = get_config()
    conn = _get_conn()
    _check_embedding_dim(conn, cfg.embed_dim)
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run()
