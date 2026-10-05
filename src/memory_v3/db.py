"""
memory-v3 -- Database Layer (sqlite-vec + FTS5)
Extended from v2: governance layers, sensory buffer, write queue,
action logging, consolidation tracking, and verification status.
"""

import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import sqlite_vec

from .config import get_config
from .retrieval.filters import (
    RetrievalFilter,
    add_provenance,
    build_eligibility_where,
    filter_eligible_ids,
)

# ---------- constants ----------

EMBEDDING_DIM = int(os.environ.get("MEMORY_V3_EMBED_DIM", "768"))
SCHEMA_VERSION = 4
RECORD_KINDS = {"derived", "handoff"}
APP_TABLES = {
    "memories",
    "memory_fts",
    "graveyard",
    "agent_offsets",
    "file_hashes",
    "conflicts",
    "compaction_receipts",
    "sensory_buffer",
    "cluster_centroids",
    "action_log",
    "consolidation_log",
    "write_queue",
    "schema_version",
    "memory_vec",
}
APP_TRIGGERS = {"memories_ai", "memories_ad", "memories_au"}
APP_INDEXES = {
    "idx_memories_type",
    "idx_memories_archived",
    "idx_memories_protected",
    "idx_memories_source",
    "idx_memories_created",
    "idx_memories_activation",
    "idx_memories_governance",
    "idx_memories_memory_layer",
    "idx_memories_cluster",
    "idx_memories_verification",
    "idx_sensory_processed",
    "idx_write_queue_status",
    "idx_action_log_action",
    "idx_memories_owner",
    "idx_memories_namespace",
    "idx_memories_owner_namespace",
    "idx_memories_source_id",
    "idx_memories_chunk_id",
    "idx_memories_expiry_at",
}
V4_MEMORY_COLUMNS = {
    "owner",
    "namespace",
    "sensitivity",
    "record_kind",
    "source_id",
    "chunk_id",
    "chunk_index",
    "chunk_count",
    "source_updated_at",
    "expiry_at",
    "project_status",
    "project_updated_at",
}
FORBIDDEN_V5_MEMORY_COLUMNS = {
    "source_generation_id",
    "source_generation_status",
    "line_start",
    "line_end",
    "heading_path",
    "rank_weight",
    "rank_marker",
    "historical",
    "deleted_at",
}
FORBIDDEN_V5_FILE_HASH_COLUMNS = {
    "source_id",
    "active_generation_id",
    "deleted_at",
}
FORBIDDEN_V5_SCHEMA_OBJECTS = {
    "source_generations",
    "staged_source_chunks",
    "indexing_checkpoints",
    "indexing_debt",
}


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to raw bytes for sqlite-vec."""
    return np.array(vec, dtype=np.float32).tobytes()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_record_kind(record_kind: str | None) -> None:
    if record_kind is not None and record_kind not in RECORD_KINDS:
        raise ValueError("record_kind must be one of: derived, handoff")


def _utc_iso(value: datetime | str | None, *, default_now: bool = False) -> str | None:
    if value is None:
        return datetime.now(timezone.utc).isoformat() if default_now else None
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC timezone")
    if parsed.utcoffset() != timedelta(0):
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


def _utc_epoch(value: datetime | str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, datetime) or parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc).timestamp()


def _register_time_functions(conn: sqlite3.Connection) -> None:
    conn.create_function("memory_v3_utc_epoch", 1, _utc_epoch, deterministic=True)


def _search_where(
    retrieval_filter: RetrievalFilter | None,
    *,
    content_type: str | None = None,
    governance_layer: int | None = None,
    alias: str = "m",
) -> tuple[str, list]:
    where_sql, params = build_eligibility_where(
        _require_owner_filter(retrieval_filter), alias=alias
    )
    clauses = [where_sql]
    if content_type:
        clauses.append(f"{alias}.content_type = ?")
        params.append(content_type)
    if governance_layer is not None:
        clauses.append(f"{alias}.governance_layer = ?")
        params.append(governance_layer)
    return " AND ".join(clauses), params


def _require_owner_filter(retrieval_filter: RetrievalFilter | None) -> RetrievalFilter:
    if retrieval_filter is None or not retrieval_filter.owner:
        raise ValueError("retrieval owner is required")
    return retrieval_filter


def _filter_search_ids(
    conn: sqlite3.Connection,
    ids: list[int],
    retrieval_filter: RetrievalFilter | None,
    *,
    content_type: str | None = None,
    governance_layer: int | None = None,
) -> set[int]:
    if not ids:
        return set()
    retrieval_filter = _require_owner_filter(retrieval_filter)
    eligible = filter_eligible_ids(conn, ids, retrieval_filter)
    if not eligible or (content_type is None and governance_layer is None):
        return eligible
    where_sql, params = _search_where(
        retrieval_filter,
        content_type=content_type,
        governance_layer=governance_layer,
        alias="m",
    )
    placeholders = ", ".join("?" for _ in eligible)
    rows = conn.execute(
        f"SELECT m.id FROM memories m WHERE m.id IN ({placeholders}) AND {where_sql}",
        (*eligible, *params),
    ).fetchall()
    return {row["id"] for row in rows}


# ---------- connection factory ----------


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with sqlite-vec loaded."""
    if db_path is None:
        db_path = get_config().db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------- schema ----------

SCHEMA_SQL = """
-- Core memories table (v2 columns + v3 extensions)
CREATE TABLE IF NOT EXISTS memories (
    -- v2 core columns
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_type TEXT DEFAULT 'fact',
    source_file TEXT,
    source_line INTEGER,
    author TEXT DEFAULT 'claude-code',
    authority_level INTEGER DEFAULT 2,
    confidence REAL DEFAULT 0.95,
    protected INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 1,
    activation_score REAL DEFAULT 0.0,
    importance_score REAL DEFAULT 0.5,
    decay_rate REAL DEFAULT 0.1,
    archived INTEGER DEFAULT 0,
    supersedes INTEGER,
    content_hash TEXT,

    -- v3 new columns
    governance_layer INTEGER DEFAULT 3,
    confidence_floor REAL DEFAULT 0.0,
    last_verified_at TEXT,
    verification_status TEXT DEFAULT 'unverified',
    cross_ref_count INTEGER DEFAULT 0,
    structure_type TEXT DEFAULT 'narrative',
    surprise_score REAL DEFAULT 0.5,
    memory_layer TEXT DEFAULT 'STM',
    cluster_id INTEGER,
    linked_memories TEXT DEFAULT '[]',
    link_descriptions TEXT DEFAULT '{}',
    source_hash TEXT,

    -- v4 metadata columns
    owner TEXT,
    namespace TEXT,
    sensitivity TEXT,
    record_kind TEXT,
    source_id TEXT,
    chunk_id TEXT,
    chunk_index INTEGER,
    chunk_count INTEGER,
    source_updated_at TEXT,
    expiry_at TEXT,
    project_status TEXT,
    project_updated_at TEXT

);

-- FTS5 full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    tags,
    source_file,
    content='memories',
    content_rowid='id'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content, tags, source_file)
    VALUES (new.id, new.content, new.tags, new.source_file);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags, source_file)
    VALUES ('delete', old.id, old.content, old.tags, old.source_file);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags, source_file)
    VALUES ('delete', old.id, old.content, old.tags, old.source_file);
    INSERT INTO memory_fts(rowid, content, tags, source_file)
    VALUES (new.id, new.content, new.tags, new.source_file);
END;

-- Graveyard for archived memories
CREATE TABLE IF NOT EXISTS graveyard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    archived_at TEXT NOT NULL,
    reason TEXT,
    last_activation_score REAL
);

-- Consumer offsets for multi-agent sync
CREATE TABLE IF NOT EXISTS agent_offsets (
    agent_id TEXT PRIMARY KEY,
    last_read_line INTEGER DEFAULT 0,
    last_read_time TEXT
);

-- File hash registry (for incremental indexing)
CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0
);

-- Conflict log
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id_a INTEGER,
    memory_id_b INTEGER,
    agent_a TEXT,
    agent_b TEXT,
    description TEXT,
    resolved INTEGER DEFAULT 0,
    resolved_by TEXT,
    created_at TEXT NOT NULL
);

-- Compaction receipts
CREATE TABLE IF NOT EXISTS compaction_receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    original_tokens INTEGER,
    compressed_tokens INTEGER,
    ratio REAL,
    protected_items_extracted INTEGER,
    verification_score REAL,
    vault_files_updated TEXT,
    receipt_data TEXT
);

-- v3: Sensory buffer (pre-encoding gate)
CREATE TABLE IF NOT EXISTS sensory_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_type TEXT DEFAULT 'fact',
    source TEXT,
    author TEXT DEFAULT 'claude-code',
    topic_cluster TEXT,
    novelty_score REAL DEFAULT 0.5,
    received_at TEXT NOT NULL,
    processed INTEGER DEFAULT 0,
    processed_at TEXT,
    decision TEXT,
    decision_reason TEXT
);

-- v3 dormant compatibility table; active v4 indexing/retrieval does not depend on it
CREATE TABLE IF NOT EXISTS cluster_centroids (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL UNIQUE,
    centroid_embedding BLOB,
    member_count INTEGER DEFAULT 0,
    label TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- v3: Action log (extraction decisions: ADD/UPDATE/DELETE/NONE)
CREATE TABLE IF NOT EXISTS action_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    new_fact TEXT NOT NULL,
    similar_ids TEXT DEFAULT '[]',
    action TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    model_version TEXT
);

-- v3: Consolidation log (merge/split/prune cycles)
CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cycle_type TEXT NOT NULL,
    stats TEXT NOT NULL
);

-- v3: Async write queue
CREATE TABLE IF NOT EXISTS write_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    payload TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

-- v3: Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    migrated_at TEXT NOT NULL,
    notes TEXT
);

-- v2 Indexes
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(content_type);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_protected ON memories(protected);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_file);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_activation ON memories(activation_score);

-- v3 Indexes
CREATE INDEX IF NOT EXISTS idx_memories_governance ON memories(governance_layer);
CREATE INDEX IF NOT EXISTS idx_memories_memory_layer ON memories(memory_layer);
CREATE INDEX IF NOT EXISTS idx_memories_cluster ON memories(cluster_id);
CREATE INDEX IF NOT EXISTS idx_memories_verification ON memories(verification_status);
CREATE INDEX IF NOT EXISTS idx_sensory_processed ON sensory_buffer(processed);
CREATE INDEX IF NOT EXISTS idx_write_queue_status ON write_queue(status, priority);
CREATE INDEX IF NOT EXISTS idx_action_log_action ON action_log(action);

-- v4 Indexes
CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_owner_namespace ON memories(owner, namespace);
CREATE INDEX IF NOT EXISTS idx_memories_source_id ON memories(source_id);
CREATE INDEX IF NOT EXISTS idx_memories_chunk_id ON memories(chunk_id);
CREATE INDEX IF NOT EXISTS idx_memories_expiry_at ON memories(expiry_at);

"""


def init_vec_table(conn: sqlite3.Connection):
    """Create the vec0 virtual table for vector search."""
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)
    conn.commit()


def _memory_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}


def _memory_indexes(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA index_list(memories)").fetchall()}


def _schema_complete(conn: sqlite3.Connection) -> bool:
    memory_columns = _memory_columns(conn)
    file_hash_columns = _table_columns(conn, "file_hashes")
    schema_objects = _app_object_names(conn)
    return (
        V4_MEMORY_COLUMNS <= memory_columns
        and FORBIDDEN_V5_MEMORY_COLUMNS.isdisjoint(memory_columns)
        and FORBIDDEN_V5_FILE_HASH_COLUMNS.isdisjoint(file_hash_columns)
        and FORBIDDEN_V5_SCHEMA_OBJECTS.isdisjoint(schema_objects)
        and APP_INDEXES <= _all_indexes(conn)
    )


def _all_indexes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    return {row[0] for row in rows if not row[0].startswith("sqlite_")}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _app_object_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
    return {row[0] for row in rows}


def _table_sql(conn: sqlite3.Connection, table: str) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return "" if row is None or row[0] is None else row[0].lower()


def _latest_schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1").fetchone()
    return None if row is None else row[0]


def _complete_v4_schema(conn: sqlite3.Connection) -> bool:
    objects = _app_object_names(conn)
    if not APP_TABLES <= objects:
        return False
    if not APP_TRIGGERS <= objects:
        return False
    if not _schema_complete(conn):
        return False
    fts_sql = _table_sql(conn, "memory_fts")
    if "using fts5" not in fts_sql or "content='memories'" not in fts_sql:
        return False
    vec_sql = _table_sql(conn, "memory_vec")
    if "using vec0" not in vec_sql or f"float[{EMBEDDING_DIM}]" not in vec_sql:
        return False
    if _latest_schema_version(conn) != SCHEMA_VERSION:
        return False
    return conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def _classify_existing_db(db_path: str) -> tuple[str, str | None]:
    if not os.path.exists(db_path):
        return "fresh", None
    if os.path.getsize(db_path) == 0:
        return "fresh", None

    uri = Path(db_path).absolute().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, timeout=10, uri=True)
    try:
        objects = _app_object_names(conn)
        if not objects:
            return "fresh", None
        has_memories = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memories'"
        ).fetchone()
        if not has_memories:
            return (
                "blocked",
                "Existing database has non-memory schema objects; run explicit migration first",
            )
        columns = _memory_columns(conn)
        if "governance_layer" not in columns:
            return "blocked", "Existing database is not v3-shaped; run explicit migration first"
        if not _complete_v4_schema(conn):
            return (
                "blocked",
                "Existing database is incomplete/version-mismatched; run explicit migration first",
            )
        return "complete", None
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialize the database with the current schema."""
    if db_path is None:
        db_path = get_config().db_path
    classification, error = _classify_existing_db(db_path)
    if classification == "blocked":
        raise RuntimeError(error)

    conn = get_connection(db_path)
    if classification == "complete":
        return conn

    conn.executescript(SCHEMA_SQL)
    init_vec_table(conn)

    if not _schema_complete(conn):
        conn.close()
        raise RuntimeError("Database schema initialization did not complete")

    # Record schema version if not already present
    existing = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    if existing == 0:
        conn.execute(
            "INSERT INTO schema_version (version, migrated_at, notes) VALUES (?, ?, ?)",
            (SCHEMA_VERSION, _now_iso(), "Fresh v4 init"),
        )
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


# ---------- CRUD operations ----------


def add_memory(
    conn: sqlite3.Connection,
    content: str,
    embedding: list[float],
    content_type: str = "fact",
    source_file: str | None = None,
    source_line: int | None = None,
    author: str = "claude-code",
    authority_level: int = 2,
    confidence: float = 0.95,
    protected: bool = False,
    tags: list[str] | None = None,
    importance_score: float = 0.5,
    decay_rate: float = 0.1,
    supersedes: int | None = None,
    # v3 new params
    governance_layer: int = 3,
    structure_type: str = "narrative",
    surprise_score: float = 0.5,
    source_hash: str | None = None,
    owner: str | None = None,
    namespace: str | None = None,
    sensitivity: str | None = None,
    record_kind: str | None = None,
    source_id: str | None = None,
    chunk_id: str | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
    source_updated_at: str | None = None,
    expiry_at: str | None = None,
    project_status: str | None = None,
    project_updated_at: str | None = None,
) -> int:
    """Add a new memory with embedding. Returns the memory ID."""
    _validate_record_kind(record_kind)
    now = _now_iso()
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    tags_json = json.dumps(tags or [])

    # Auto-determine governance layer from protected/type if not explicitly set
    if protected and governance_layer == 3:
        governance_layer = 1  # Protected memories are constitutional

    cursor = conn.execute(
        """INSERT INTO memories
        (content, content_type, source_file, source_line, author,
         authority_level, confidence, protected, tags,
         created_at, updated_at, last_accessed_at,
         importance_score, decay_rate, supersedes, content_hash,
         governance_layer, structure_type, surprise_score, source_hash,
         owner, namespace, sensitivity, record_kind, source_id, chunk_id,
         chunk_index, chunk_count, source_updated_at, expiry_at,
         project_status, project_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            content,
            content_type,
            source_file,
            source_line,
            author,
            authority_level,
            confidence,
            1 if protected else 0,
            tags_json,
            now,
            now,
            now,
            importance_score,
            decay_rate,
            supersedes,
            content_hash,
            governance_layer,
            structure_type,
            surprise_score,
            source_hash,
            owner,
            namespace,
            sensitivity,
            record_kind,
            source_id,
            chunk_id,
            chunk_index,
            chunk_count,
            source_updated_at,
            expiry_at,
            project_status,
            project_updated_at,
        ),
    )
    memory_id = cursor.lastrowid

    # Insert embedding into vec0
    conn.execute(
        "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
        (memory_id, _serialize_f32(embedding)),
    )
    conn.commit()
    return memory_id


def replace_source_chunks(
    conn: sqlite3.Connection,
    *,
    owner: str,
    namespace: str,
    source_id: str,
    source_path: str,
    source_hash: str,
    source_mtime: str,
    chunks: list[dict],
    content_type: str,
    tags: list[str],
    governance_layer: int,
    project_status: str | None,
    project_updated_at: str | None,
    embed_model: str,
    embed_provider: str,
    embed_dim: int,
) -> dict:
    if len(chunks) == 0:
        raise ValueError("replacement requires at least one chunk")
    for index, chunk in enumerate(chunks):
        if chunk["chunk_index"] != index:
            raise ValueError("chunk indexes must be contiguous")
        if chunk["chunk_count"] != len(chunks):
            raise ValueError("chunk count mismatch")
        _validate_embedding_blob(chunk["embedding"], embed_dim)

    tags_json = json.dumps(tags or [])
    now = _now_iso()
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        old_rows = conn.execute(
            """
            SELECT id FROM memories
            WHERE owner = ? AND source_id = ? AND record_kind = 'derived'
            ORDER BY id
            """,
            (owner, source_id),
        ).fetchall()
        old_ids = [row["id"] for row in old_rows]
        for memory_id in old_ids:
            conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
        if old_ids:
            placeholders = ", ".join("?" for _ in old_ids)
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", old_ids)
        _call_replace_hook(conn, "after_delete")

        inserted_ids: list[int] = []
        for chunk in chunks:
            cursor = conn.execute(
                """INSERT INTO memories
                (content, content_type, source_file, source_line, author,
                 authority_level, confidence, protected, tags,
                 created_at, updated_at, last_accessed_at,
                 importance_score, decay_rate, supersedes, content_hash,
                 governance_layer, structure_type, surprise_score, source_hash,
                 owner, namespace, sensitivity, record_kind, source_id, chunk_id,
                 chunk_index, chunk_count, source_updated_at, expiry_at,
                 project_status, project_updated_at)
                VALUES (?, ?, ?, ?, 'memory-v3-indexer', 2, 0.95, 0, ?,
                        ?, ?, ?, 0.5, 0.1, NULL, ?, ?, 'narrative', 0.5, ?,
                        ?, ?, 'normal', 'derived', ?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    chunk["content"],
                    content_type,
                    source_path,
                    chunk["source_line"],
                    tags_json,
                    now,
                    now,
                    now,
                    hashlib.sha256(chunk["content"].encode()).hexdigest()[:16],
                    governance_layer,
                    source_hash,
                    owner,
                    namespace,
                    source_id,
                    chunk["chunk_id"],
                    chunk["chunk_index"],
                    len(chunks),
                    source_mtime,
                    project_status,
                    project_updated_at,
                ),
            )
            memory_id = cursor.lastrowid
            inserted_ids.append(memory_id)
            conn.execute(
                "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                (memory_id, chunk["embedding"]),
            )
            if len(inserted_ids) == 1:
                _call_replace_hook(conn, "after_partial_insert")

        _call_replace_hook(conn, "before_file_hash")
        conn.execute(
            """
            INSERT INTO file_hashes (file_path, content_hash, indexed_at, chunk_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                content_hash = excluded.content_hash,
                indexed_at = excluded.indexed_at,
                chunk_count = excluded.chunk_count
            """,
            (source_path, source_hash, now, len(chunks)),
        )
        _call_replace_hook(conn, "before_commit")
        conn.commit()
        return {"deleted": len(old_ids), "inserted": len(inserted_ids), "chunk_count": len(chunks)}
    except Exception:
        conn.rollback()
        raise


def cleanup_source(
    conn: sqlite3.Connection,
    *,
    owner: str,
    source_id: str,
    file_path: str | None = None,
) -> dict:
    if not owner:
        raise ValueError("owner is required")
    if not source_id:
        raise ValueError("source_id is required")
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        clauses = ["owner = ?", "source_id = ?", "record_kind = 'derived'"]
        params: list[str] = [owner, source_id]
        if file_path is not None:
            clauses.append("source_file = ?")
            params.append(file_path)
        rows = conn.execute(
            f"SELECT id, source_file FROM memories WHERE {' AND '.join(clauses)} ORDER BY id",
            params,
        ).fetchall()
        memory_ids = [row["id"] for row in rows]
        paths = {row["source_file"] for row in rows if row["source_file"] is not None}
        for memory_id in memory_ids:
            conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
        if memory_ids:
            placeholders = ", ".join("?" for _ in memory_ids)
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", memory_ids)
        deleted_hashes = 0
        for path in sorted(paths):
            survivor = conn.execute(
                """
                SELECT 1 FROM memories
                WHERE source_file = ? AND record_kind = 'derived' AND archived = 0
                LIMIT 1
                """,
                (path,),
            ).fetchone()
            if survivor is None:
                cursor = conn.execute("DELETE FROM file_hashes WHERE file_path = ?", (path,))
                deleted_hashes += cursor.rowcount
        hook = getattr(cleanup_source, "test_hook", None)
        if hook is not None:
            hook(conn, source_id, memory_ids)
        conn.commit()
        return {
            "deleted_memories": len(memory_ids),
            "deleted_vectors": len(memory_ids),
            "deleted_file_hashes": deleted_hashes,
        }
    except Exception:
        conn.rollback()
        raise


def _validate_embedding_blob(blob: bytes, expected_dim: int) -> None:
    if not isinstance(blob, bytes) or len(blob) != expected_dim * 4:
        raise ValueError("invalid embedding length")
    array = np.frombuffer(blob, dtype=np.float32)
    if array.shape != (expected_dim,) or not np.isfinite(array).all():
        raise ValueError("invalid embedding finite values")


def _call_replace_hook(conn: sqlite3.Connection, phase: str) -> None:
    hook = getattr(replace_source_chunks, "test_hook", None)
    if hook is not None:
        hook(conn, phase)


def list_active_handoffs(
    conn: sqlite3.Connection,
    agent_id: str | None = None,
    since: datetime | str | None = None,
    now: datetime | str | None = None,
) -> list[dict]:
    now_iso = _utc_iso(now, default_now=True)
    now_epoch = _utc_epoch(now_iso)
    retrieval_filter = RetrievalFilter(
        owner=get_config().source_policy_owner,
        record_kind="handoff",
    )
    where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
    clauses = [
        where_sql,
        "memory_v3_utc_epoch(m.created_at) IS NOT NULL",
        "memory_v3_utc_epoch(m.expiry_at) > ?",
    ]
    params.append(now_epoch)
    if agent_id is not None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        clauses.append("m.author = ?")
        params.append(agent_id.strip())
    since_iso = _utc_iso(since)
    if since_iso is not None:
        clauses.append("memory_v3_utc_epoch(m.created_at) >= ?")
        params.append(_utc_epoch(since_iso))
    _register_time_functions(conn)
    rows = conn.execute(
        f"""SELECT m.* FROM memories m
        WHERE {" AND ".join(clauses)}
        ORDER BY memory_v3_utc_epoch(m.created_at), m.id""",
        params,
    ).fetchall()
    return [add_provenance(dict(row)) for row in rows]


def sweep_expired_handoffs(
    conn: sqlite3.Connection,
    now: datetime | str | None = None,
) -> int:
    cfg = get_config()
    if not cfg.handoff_sweeper_enabled:
        return 0
    now_iso = _utc_iso(now, default_now=True)
    now_epoch = _utc_epoch(now_iso)
    _register_time_functions(conn)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            """
            SELECT id FROM memories
            WHERE owner = ?
              AND record_kind = 'handoff'
              AND expiry_at IS NOT NULL
              AND memory_v3_utc_epoch(expiry_at) IS NOT NULL
              AND memory_v3_utc_epoch(expiry_at) <= ?
            ORDER BY id
            """,
            (cfg.source_policy_owner, now_epoch),
        ).fetchall()
        memory_ids = [row["id"] for row in rows]
        for memory_id in memory_ids:
            conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
        if memory_ids:
            placeholders = ", ".join("?" for _ in memory_ids)
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", memory_ids)
            _scrub_linked_memories(conn, cfg.source_policy_owner, memory_ids)
        hook = getattr(sweep_expired_handoffs, "test_hook", None)
        if hook is not None:
            hook(conn, memory_ids)
        conn.commit()
        return len(memory_ids)
    except Exception:
        conn.rollback()
        raise


def _scrub_linked_memories(conn: sqlite3.Connection, owner: str, deleted_ids: list[int]) -> int:
    deleted = {str(memory_id) for memory_id in deleted_ids}
    if not deleted:
        return 0
    updated = 0
    rows = conn.execute(
        "SELECT id, linked_memories FROM memories WHERE owner = ?",
        (owner,),
    ).fetchall()
    for row in rows:
        raw = row["linked_memories"]
        if not raw:
            continue
        try:
            links = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(links, list):
            continue
        kept = []
        changed = False
        for link in links:
            memory_id = link.get("memory_id") if isinstance(link, dict) else link
            if str(memory_id) in deleted:
                changed = True
                continue
            kept.append(link)
        if changed:
            conn.execute(
                "UPDATE memories SET linked_memories = ?, updated_at = ? WHERE id = ?",
                (json.dumps(kept), _now_iso(), row["id"]),
            )
            updated += 1
    return updated


def get_memory(conn: sqlite3.Connection, memory_id: int) -> dict | None:
    """Retrieve a single memory by ID. Updates access timestamp."""
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND archived = 0", (memory_id,)
    ).fetchone()
    if row is None:
        return None

    # Update access stats
    now = _now_iso()
    conn.execute(
        """UPDATE memories
        SET last_accessed_at = ?, access_count = access_count + 1
        WHERE id = ?""",
        (now, memory_id),
    )
    conn.commit()
    return dict(row)


def update_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    content: str | None = None,
    tags: list[str] | None = None,
    protected: bool | None = None,
    importance_score: float | None = None,
    # v3 new params
    governance_layer: int | None = None,
    structure_type: str | None = None,
    verification_status: str | None = None,
    confidence: float | None = None,
    memory_layer: str | None = None,
    cluster_id: int | None = None,
    linked_memories: list[int] | None = None,
    link_descriptions: dict | None = None,
    cross_ref_count: int | None = None,
    surprise_score: float | None = None,
    owner: str | None = None,
    namespace: str | None = None,
    sensitivity: str | None = None,
    record_kind: str | None = None,
    source_id: str | None = None,
    chunk_id: str | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
    source_updated_at: str | None = None,
    expiry_at: str | None = None,
    project_status: str | None = None,
    project_updated_at: str | None = None,
) -> bool:
    """Update an existing memory. Only provided fields are changed."""
    _validate_record_kind(record_kind)
    updates = []
    params = []

    if content is not None:
        updates.append("content = ?")
        params.append(content)
        updates.append("content_hash = ?")
        params.append(hashlib.sha256(content.encode()).hexdigest()[:16])
    if tags is not None:
        updates.append("tags = ?")
        params.append(json.dumps(tags))
    if protected is not None:
        updates.append("protected = ?")
        params.append(1 if protected else 0)
    if importance_score is not None:
        updates.append("importance_score = ?")
        params.append(importance_score)
    if governance_layer is not None:
        updates.append("governance_layer = ?")
        params.append(governance_layer)
    if structure_type is not None:
        updates.append("structure_type = ?")
        params.append(structure_type)
    if verification_status is not None:
        updates.append("verification_status = ?")
        params.append(verification_status)
        if verification_status == "verified":
            updates.append("last_verified_at = ?")
            params.append(_now_iso())
    if confidence is not None:
        updates.append("confidence = ?")
        params.append(confidence)
    if memory_layer is not None:
        updates.append("memory_layer = ?")
        params.append(memory_layer)
    if cluster_id is not None:
        updates.append("cluster_id = ?")
        params.append(cluster_id)
    if linked_memories is not None:
        updates.append("linked_memories = ?")
        params.append(json.dumps(linked_memories))
    if link_descriptions is not None:
        updates.append("link_descriptions = ?")
        params.append(json.dumps(link_descriptions))
    if cross_ref_count is not None:
        updates.append("cross_ref_count = ?")
        params.append(cross_ref_count)
    if surprise_score is not None:
        updates.append("surprise_score = ?")
        params.append(surprise_score)
    metadata_updates = {
        "owner": owner,
        "namespace": namespace,
        "sensitivity": sensitivity,
        "record_kind": record_kind,
        "source_id": source_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "source_updated_at": source_updated_at,
        "expiry_at": expiry_at,
        "project_status": project_status,
        "project_updated_at": project_updated_at,
    }
    for column, value in metadata_updates.items():
        if value is not None:
            updates.append(f"{column} = ?")
            params.append(value)

    if not updates:
        return False

    updates.append("updated_at = ?")
    params.append(_now_iso())
    params.append(memory_id)

    conn.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    return True


def archive_memory(conn: sqlite3.Connection, memory_id: int, reason: str = "decay") -> bool:
    """Archive a memory to the graveyard. Respects governance layer rules."""
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND archived = 0", (memory_id,)
    ).fetchone()
    if row is None:
        return False

    mem = dict(row)

    # Never archive protected memories
    if mem["protected"]:
        return False

    # Never archive constitutional or legislative layers
    if mem.get("governance_layer", 3) in (1, 2):
        return False

    # Write to graveyard
    conn.execute(
        """INSERT INTO graveyard
        (memory_id, content, metadata, archived_at, reason, last_activation_score)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (memory_id, mem["content"], json.dumps(mem), _now_iso(), reason, mem["activation_score"]),
    )

    # Mark as archived
    conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,))

    # Remove from vec index
    conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
    conn.commit()
    return True


# ---------- Search ----------


def hybrid_search(
    conn: sqlite3.Connection,
    query_text: str,
    query_embedding: list[float],
    limit: int = 10,
    content_type: str | None = None,
    include_archived: bool = False,
    governance_layer: int | None = None,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[dict]:
    """
    Hybrid search: BM25 (FTS5) + vector similarity (vec0).
    Uses Reciprocal Rank Fusion to combine results.
    """
    retrieval_filter = _require_owner_filter(retrieval_filter)
    cfg = get_config()
    k = cfg.rrf_k
    where_sql, params = _search_where(
        retrieval_filter,
        content_type=content_type,
        governance_layer=governance_layer,
        alias="m",
    )
    eligible_rows = conn.execute(
        f"SELECT m.id FROM memories m WHERE {where_sql} ORDER BY m.id",
        params,
    ).fetchall()
    eligible_ids = [row["id"] for row in eligible_rows]

    # --- BM25 keyword search ---
    fts_results = {}
    try:
        rows = conn.execute(
            f"""SELECT memory_fts.rowid, rank FROM memory_fts
            JOIN memories m ON m.id = memory_fts.rowid
            WHERE memory_fts MATCH ? AND {where_sql}
            ORDER BY rank, memory_fts.rowid LIMIT ?""",
            (query_text, *params, limit * 3),
        ).fetchall()
        for rank_pos, row in enumerate(rows):
            fts_results[row["rowid"]] = 1.0 / (k + rank_pos + 1)
    except Exception:
        pass  # FTS match syntax errors are non-fatal

    # --- Vector similarity search ---
    vec_results = {}
    vec_rows = []
    query_array = np.asarray(query_embedding, dtype=np.float32)
    batch_size = 500
    for i in range(0, len(eligible_ids), batch_size):
        batch = eligible_ids[i : i + batch_size]
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT id, embedding FROM memory_vec WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        for row in rows:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            distance = float(np.linalg.norm(embedding - query_array))
            vec_rows.append((row["id"], distance))
    vec_rows.sort(key=lambda row: (row[1], row[0]))
    for rank_pos, (memory_id, _distance) in enumerate(vec_rows[: max(limit * 3, limit)]):
        vec_results[memory_id] = 1.0 / (k + rank_pos + 1)

    # --- Reciprocal Rank Fusion ---
    all_ids = set(fts_results.keys()) | set(vec_results.keys())
    fused = []
    for mid in all_ids:
        score = fts_results.get(mid, 0) + vec_results.get(mid, 0)
        fused.append((mid, score))

    fused.sort(key=lambda x: (-x[1], x[0]))

    # Fetch full records
    results = []
    for mid, score in fused[: limit * 2]:
        row = conn.execute(
            f"SELECT m.* FROM memories m WHERE m.id = ? AND {where_sql}", (mid, *params)
        ).fetchone()
        if row:
            mem = add_provenance(dict(row))
            mem["hybrid_score"] = score
            mem["bm25_score"] = fts_results.get(mid, 0)
            mem["vec_score"] = vec_results.get(mid, 0)
            results.append(mem)

        if len(results) >= limit:
            break

    # Update access timestamps for returned results
    now = _now_iso()
    for mem in results:
        conn.execute(
            """UPDATE memories
            SET last_accessed_at = ?, access_count = access_count + 1
            WHERE id = ?""",
            (now, mem["id"]),
        )
    conn.commit()

    return results


def keyword_search(
    conn: sqlite3.Connection,
    keywords: str,
    limit: int = 10,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[dict]:
    """Pure BM25 keyword search via FTS5."""
    retrieval_filter = _require_owner_filter(retrieval_filter)
    try:
        where_sql, params = _search_where(retrieval_filter, alias="m")
        rows = conn.execute(
            f"""SELECT memory_fts.rowid, rank FROM memory_fts
            JOIN memories m ON m.id = memory_fts.rowid
            WHERE memory_fts MATCH ? AND {where_sql}
            ORDER BY rank LIMIT ?""",
            (keywords, *params, limit),
        ).fetchall()
    except Exception:
        return []

    results = []
    now = _now_iso()
    for row in rows:
        mem_row = conn.execute("SELECT * FROM memories WHERE id = ?", (row["rowid"],)).fetchone()
        if mem_row:
            mem = add_provenance(dict(mem_row))
            mem["bm25_rank"] = row["rank"]
            results.append(mem)
            conn.execute(
                """UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE id = ?""",
                (now, mem["id"]),
            )
    conn.commit()
    return results


def list_recent(
    conn: sqlite3.Connection,
    hours: int = 24,
    limit: int = 20,
    now: datetime | str | None = None,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[dict]:
    """Get memories created in the UTC window [now - hours, now]."""
    if (
        not isinstance(hours, (int, float))
        or isinstance(hours, bool)
        or not math.isfinite(hours)
        or hours <= 0
        or hours > 24 * 366
    ):
        raise ValueError("hours must be a positive number no greater than 8784")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer from 1 to 1000")

    now_iso = _utc_iso(now, default_now=True)
    now_epoch = _utc_epoch(now_iso)
    cutoff_epoch = now_epoch - (float(hours) * 3600)
    retrieval_filter = _require_owner_filter(retrieval_filter)
    where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
    _register_time_functions(conn)
    rows = conn.execute(
        f"""SELECT m.* FROM memories m
        WHERE {where_sql}
          AND memory_v3_utc_epoch(m.created_at) >= ?
          AND memory_v3_utc_epoch(m.created_at) <= ?
          AND (
            m.record_kind != 'handoff'
            OR (
              m.expiry_at IS NOT NULL
              AND memory_v3_utc_epoch(m.expiry_at) > ?
            )
          )
        ORDER BY memory_v3_utc_epoch(m.created_at) DESC, m.id DESC
        LIMIT ?""",
        (*params, cutoff_epoch, now_epoch, now_epoch, limit),
    ).fetchall()
    return [add_provenance(dict(r)) for r in rows]


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get database statistics including v3 governance layer counts."""
    total = conn.execute("SELECT COUNT(*) FROM memories WHERE archived = 0").fetchone()[0]
    archived = conn.execute("SELECT COUNT(*) FROM memories WHERE archived = 1").fetchone()[0]
    protected = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE protected = 1 AND archived = 0"
    ).fetchone()[0]
    graveyard_count = conn.execute("SELECT COUNT(*) FROM graveyard").fetchone()[0]
    files_indexed = conn.execute("SELECT COUNT(*) FROM file_hashes").fetchone()[0]

    # Type counts
    type_counts = {}
    for row in conn.execute(
        "SELECT content_type, COUNT(*) as cnt FROM memories "
        "WHERE archived = 0 GROUP BY content_type"
    ).fetchall():
        type_counts[row["content_type"]] = row["cnt"]

    # v3: Governance layer counts
    layer_counts = {}
    for row in conn.execute(
        "SELECT governance_layer, COUNT(*) as cnt FROM memories "
        "WHERE archived = 0 GROUP BY governance_layer"
    ).fetchall():
        layer_counts[row["governance_layer"]] = row["cnt"]

    # v3: Memory layer counts (STM/LTM)
    memory_layer_counts = {}
    for row in conn.execute(
        "SELECT memory_layer, COUNT(*) as cnt FROM memories "
        "WHERE archived = 0 GROUP BY memory_layer"
    ).fetchall():
        memory_layer_counts[row["memory_layer"]] = row["cnt"]

    # v3: Verification status counts
    verification_counts = {}
    for row in conn.execute(
        "SELECT verification_status, COUNT(*) as cnt FROM memories "
        "WHERE archived = 0 GROUP BY verification_status"
    ).fetchall():
        verification_counts[row["verification_status"]] = row["cnt"]

    # v3: Sensory buffer stats
    sensory_pending = conn.execute(
        "SELECT COUNT(*) FROM sensory_buffer WHERE processed = 0"
    ).fetchone()[0]

    # v3: Write queue stats
    write_queue_pending = conn.execute(
        "SELECT COUNT(*) FROM write_queue WHERE status = 'pending'"
    ).fetchone()[0]

    # v3: Cluster count
    cluster_count = conn.execute(
        "SELECT COUNT(DISTINCT cluster_id) FROM memories "
        "WHERE archived = 0 AND cluster_id IS NOT NULL"
    ).fetchone()[0]

    return {
        "total_active": total,
        "archived": archived,
        "protected": protected,
        "graveyard": graveyard_count,
        "files_indexed": files_indexed,
        "type_counts": type_counts,
        "governance_layer_counts": layer_counts,
        "memory_layer_counts": memory_layer_counts,
        "verification_counts": verification_counts,
        "sensory_buffer_pending": sensory_pending,
        "write_queue_pending": write_queue_pending,
        "cluster_count": cluster_count,
    }


# ---------- Sensory Buffer ----------


def add_to_sensory_buffer(
    conn: sqlite3.Connection,
    content: str,
    content_hash: str,
    content_type: str = "fact",
    source: str | None = None,
    author: str = "claude-code",
    topic_cluster: str | None = None,
    novelty_score: float = 0.5,
) -> int:
    """Add an item to the sensory buffer for pre-encoding evaluation.
    Returns the buffer item ID."""
    cursor = conn.execute(
        """INSERT INTO sensory_buffer
        (content, content_hash, content_type, source, author,
         topic_cluster, novelty_score, received_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            content,
            content_hash,
            content_type,
            source,
            author,
            topic_cluster,
            novelty_score,
            _now_iso(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_pending_sensory(
    conn: sqlite3.Connection,
    limit: int = 50,
) -> list[dict]:
    """Get unprocessed items from the sensory buffer."""
    rows = conn.execute(
        """SELECT * FROM sensory_buffer
        WHERE processed = 0
        ORDER BY received_at ASC
        LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_sensory_processed(
    conn: sqlite3.Connection,
    buffer_id: int,
    processed: int = 1,
    decision: str | None = None,
    decision_reason: str | None = None,
):
    """Mark a sensory buffer item as processed with optional decision info."""
    conn.execute(
        """UPDATE sensory_buffer
        SET processed = ?, processed_at = ?, decision = ?, decision_reason = ?
        WHERE id = ?""",
        (processed, _now_iso(), decision, decision_reason, buffer_id),
    )
    conn.commit()


# ---------- Write Queue ----------


def enqueue_write(
    conn: sqlite3.Connection,
    operation: str,
    payload: dict | str,
    priority: int = 5,
) -> int:
    """Enqueue an async write operation. Returns the write queue ID."""
    if isinstance(payload, dict):
        payload = json.dumps(payload)
    cursor = conn.execute(
        """INSERT INTO write_queue (operation, payload, priority, created_at)
        VALUES (?, ?, ?, ?)""",
        (operation, payload, priority, _now_iso()),
    )
    conn.commit()
    return cursor.lastrowid


def get_pending_writes(
    conn: sqlite3.Connection,
    limit: int = 10,
) -> list[dict]:
    """Get pending write operations ordered by priority (lower = higher priority)."""
    rows = conn.execute(
        """SELECT * FROM write_queue
        WHERE status = 'pending'
        ORDER BY priority ASC, created_at ASC
        LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_write_complete(
    conn: sqlite3.Connection,
    write_id: int,
    status: str = "completed",
    error: str | None = None,
):
    """Mark a write queue item as completed or failed."""
    conn.execute(
        """UPDATE write_queue
        SET status = ?, error = ?, completed_at = ?
        WHERE id = ?""",
        (status, error, _now_iso(), write_id),
    )
    conn.commit()


# ---------- Action Log ----------


def log_action(
    conn: sqlite3.Connection,
    new_fact: str,
    similar_ids: list[int],
    action: str,
    confidence: float = 0.0,
    model_version: str | None = None,
) -> int:
    """Log an extraction decision (ADD/UPDATE/DELETE/NONE). Returns log entry ID."""
    cursor = conn.execute(
        """INSERT INTO action_log
        (timestamp, new_fact, similar_ids, action, confidence, model_version)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (_now_iso(), new_fact, json.dumps(similar_ids), action, confidence, model_version),
    )
    conn.commit()
    return cursor.lastrowid


# ---------- Consolidation Log ----------


def log_consolidation(
    conn: sqlite3.Connection,
    cycle_type: str,
    stats: dict,
) -> int:
    """Log a consolidation cycle (merge/split/prune). Returns log entry ID."""
    cursor = conn.execute(
        """INSERT INTO consolidation_log (timestamp, cycle_type, stats)
        VALUES (?, ?, ?)""",
        (_now_iso(), cycle_type, json.dumps(stats)),
    )
    conn.commit()
    return cursor.lastrowid
