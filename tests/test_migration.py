"""
Tests for memory_v3.migration.

Covers v2 detection, v2->latest migration, v3->v4 metadata upgrade,
idempotency, preservation of search/vector linkage, and rollback.
"""

import sqlite3

import pytest
import sqlite_vec

from memory_v3 import db
from memory_v3.migration import (
    V3_NEW_COLUMNS,
    V4_INDEXES_SQL,
    V4_NEW_COLUMNS,
    detect_v2_db,
    migrate_v2_to_v3,
    migrate_v3_to_v4,
)

V5_ONLY_MEMORY_COLUMNS = {
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

V5_ONLY_FILE_HASH_COLUMNS = {"source_id", "active_generation_id", "deleted_at"}

V5_ONLY_TABLES = {
    "source_generations",
    "staged_source_chunks",
    "indexing_checkpoints",
    "indexing_debt",
}

V2_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
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
    content_hash TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content, tags, source_file,
    content='memories', content_rowid='id'
);

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

CREATE TABLE IF NOT EXISTS graveyard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    archived_at TEXT NOT NULL,
    reason TEXT,
    last_activation_score REAL
);

CREATE TABLE IF NOT EXISTS agent_offsets (
    agent_id TEXT PRIMARY KEY,
    last_read_line INTEGER DEFAULT 0,
    last_read_time TEXT
);

CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0
);

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

CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[768]
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(content_type);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_protected ON memories(protected);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_file);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_activation ON memories(activation_score);
"""


def _connect(db_path) -> sqlite3.Connection:
    assert db_path.is_relative_to(db_path.parent)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def _make_v2_db(tmp_path) -> tuple[sqlite3.Connection, object]:
    db_path = tmp_path / "legacy-v2.db"
    conn = _connect(db_path)
    conn.executescript(V2_SCHEMA_SQL)
    conn.commit()
    return conn, db_path


def _make_v3_db(tmp_path) -> tuple[sqlite3.Connection, object]:
    conn, db_path = _make_v2_db(tmp_path)
    migrate_v2_to_v3(conn)
    return conn, db_path


def _make_v4_db(tmp_path) -> tuple[sqlite3.Connection, object]:
    conn, db_path = _make_v3_db(tmp_path)
    migrate_v3_to_v4(conn)
    return conn, db_path


def _insert_legacy_memory(conn, content, content_type="fact", protected=0):
    now = "2026-10-01T00:00:00+00:00"
    cursor = conn.execute(
        """INSERT INTO memories
        (content, content_type, protected, source_file, source_line,
         created_at, updated_at, last_accessed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, content_type, protected, "legacy.md", 7, now, now, now),
    )
    conn.commit()
    return cursor.lastrowid


def _memory_columns(conn) -> set[str]:
    return {row["name"] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}


def _index_name(statement: str) -> str:
    return statement.split(" EXISTS ", 1)[1].split(" ", 1)[0]


def _schema_snapshot(conn):
    rows = conn.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    latest = conn.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1").fetchone()
    return {
        "objects": [(row[0], row[1], row[2], row[3]) for row in rows],
        "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
        "latest_schema": None if latest is None else latest[0],
    }


class TestDetectV2Db:
    def test_detect_v2_db(self, tmp_path):
        conn, db_path = _make_v2_db(tmp_path)
        try:
            assert db_path.is_relative_to(tmp_path)
            assert detect_v2_db(conn) is True
        finally:
            conn.close()

    def test_detect_v3_db_is_not_v2(self, tmp_path):
        conn, db_path = _make_v3_db(tmp_path)
        try:
            assert db_path.is_relative_to(tmp_path)
            assert detect_v2_db(conn) is False
        finally:
            conn.close()

    def test_detect_empty_db(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = _connect(db_path)
        try:
            assert db_path.is_relative_to(tmp_path)
            assert detect_v2_db(conn) is False
        finally:
            conn.close()


class TestMigrateAddsColumns:
    def test_migrate_v2_to_v3_adds_only_v3_columns(self, tmp_path):
        conn, db_path = _make_v2_db(tmp_path)
        try:
            assert db_path.is_relative_to(tmp_path)
            migrate_v2_to_v3(conn)
            col_names = _memory_columns(conn)

            for col_name, _ in V3_NEW_COLUMNS:
                assert col_name in col_names, f"Missing column after migration: {col_name}"
            for col_name in {name for name, _ in V4_NEW_COLUMNS} | V5_ONLY_MEMORY_COLUMNS:
                assert col_name not in col_names
        finally:
            conn.close()

    def test_v3_to_v4_adds_metadata_columns_and_indexes(self, tmp_path):
        conn, db_path = _make_v3_db(tmp_path)
        try:
            assert db_path.is_relative_to(tmp_path)
            assert "owner" not in _memory_columns(conn)

            stats = migrate_v3_to_v4(conn)
            assert stats["columns_added"] == len(V4_NEW_COLUMNS)

            col_names = _memory_columns(conn)
            for col_name, _ in V4_NEW_COLUMNS:
                assert col_name in col_names

            indexes = {row["name"] for row in conn.execute("PRAGMA index_list(memories)")}
            expected_indexes = {statement.split()[5] for statement in V4_INDEXES_SQL}
            assert expected_indexes <= indexes
        finally:
            conn.close()

    def test_v3_to_v4_claims_v4_only_without_v5_objects(self, tmp_path):
        conn, db_path = _make_v3_db(tmp_path)
        try:
            assert db_path.is_relative_to(tmp_path)
            migrate_v3_to_v4(conn)

            memory_cols = _memory_columns(conn)
            file_hash_cols = {row["name"] for row in conn.execute("PRAGMA table_info(file_hashes)")}
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
            }
            assert V5_ONLY_MEMORY_COLUMNS.isdisjoint(memory_cols)
            assert V5_ONLY_FILE_HASH_COLUMNS.isdisjoint(file_hash_cols)
            assert V5_ONLY_TABLES.isdisjoint(tables)
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        finally:
            conn.close()


class TestMigrateGovernanceClassification:
    def test_migrate_governance_classification(self, tmp_path):
        conn, _ = _make_v2_db(tmp_path)
        try:
            _insert_legacy_memory(conn, "I am Sean", content_type="identity", protected=1)
            _insert_legacy_memory(
                conn,
                "We decided to use Python",
                content_type="decision",
                protected=0,
            )
            _insert_legacy_memory(conn, "Just a fact", content_type="fact", protected=0)
            _insert_legacy_memory(
                conn,
                "A conversation happened",
                content_type="episode",
                protected=0,
            )

            stats = migrate_v2_to_v3(conn)

            assert stats["constitutional_count"] >= 1
            assert stats["legislative_count"] >= 1
            assert stats["ephemeral_count"] >= 1
            row_identity = conn.execute(
                "SELECT governance_layer FROM memories WHERE content = 'I am Sean'"
            ).fetchone()
            assert row_identity[0] == 1

            row_decision = conn.execute(
                "SELECT governance_layer FROM memories WHERE content = 'We decided to use Python'"
            ).fetchone()
            assert row_decision[0] == 2

            row_episode = conn.execute(
                "SELECT governance_layer FROM memories WHERE content = 'A conversation happened'"
            ).fetchone()
            assert row_episode[0] == 4

            row_fact = conn.execute(
                "SELECT governance_layer FROM memories WHERE content = 'Just a fact'"
            ).fetchone()
            assert row_fact[0] == 3
        finally:
            conn.close()


class TestMigrateCreatesNewTables:
    V3_NEW_TABLES = [
        "sensory_buffer",
        "cluster_centroids",
        "action_log",
        "consolidation_log",
        "write_queue",
        "schema_version",
    ]

    def test_migrate_creates_new_tables(self, tmp_path):
        conn, _ = _make_v2_db(tmp_path)
        try:
            migrate_v2_to_v3(conn)

            rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            table_names = {row[0] for row in rows}

            for table in self.V3_NEW_TABLES:
                assert table in table_names, f"Missing table after migration: {table}"
        finally:
            conn.close()


class TestMigrateIdempotent:
    def test_migrate_v2_to_v3_idempotent(self, tmp_path):
        conn, _ = _make_v2_db(tmp_path)
        try:
            memory_id = _insert_legacy_memory(conn, "test memory", protected=0)

            stats1 = migrate_v2_to_v3(conn)
            assert stats1["columns_added"] == len(V3_NEW_COLUMNS)
            assert "metadata_columns_added" not in stats1
            assert "lifecycle_memory_columns_added" not in stats1

            conn.execute(
                "UPDATE memories SET governance_layer = 1 WHERE id = ?",
                (memory_id,),
            )
            conn.commit()

            stats2 = migrate_v2_to_v3(conn)
            assert stats2["columns_added"] == 0
            assert stats2["constitutional_count"] == 0
            assert stats2["legislative_count"] == 0
            assert stats2["ephemeral_count"] == 0

            governance_layer = conn.execute(
                "SELECT governance_layer FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()["governance_layer"]
            assert governance_layer == 1

            versions = conn.execute("SELECT version FROM schema_version").fetchall()
            assert [row["version"] for row in versions].count(3) == 1
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        finally:
            conn.close()

    def test_migrate_v3_to_v4_idempotent(self, tmp_path):
        conn, _ = _make_v3_db(tmp_path)
        try:
            stats1 = migrate_v3_to_v4(conn)
            stats2 = migrate_v3_to_v4(conn)
            assert stats1["columns_added"] == len(V4_NEW_COLUMNS)
            assert stats2["columns_added"] == 0
            latest = conn.execute(
                "SELECT version FROM schema_version ORDER BY id DESC LIMIT 1"
            ).fetchone()["version"]
            assert latest == 4
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        finally:
            conn.close()

    def test_full_explicit_chain_reaches_v4_and_reopens(self, tmp_path):
        conn, db_path = _make_v2_db(tmp_path)
        try:
            migrate_v2_to_v3(conn)
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
            migrate_v3_to_v4(conn)
            assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        finally:
            conn.close()

        conn = db.init_db(str(db_path))
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        finally:
            conn.close()

    @pytest.mark.parametrize(
        ("migrate", "expected_version"),
        [
            (lambda conn: migrate_v2_to_v3(conn), 3),
        ],
    )
    def test_intermediate_versions_do_not_normal_open(
        self,
        tmp_path,
        migrate,
        expected_version,
    ):
        conn, db_path = _make_v2_db(tmp_path)
        try:
            migrate(conn)
            assert conn.execute("PRAGMA user_version").fetchone()[0] == expected_version
        finally:
            conn.close()

        with pytest.raises(RuntimeError, match="run explicit migration first"):
            db.init_db(str(db_path))


class TestMigratePreservesDataAndLinks:
    def test_v3_to_v4_preserves_rows_ids_fts_vector_and_file_hashes(
        self,
        tmp_path,
        mock_embedding,
    ):
        conn, db_path = _make_v3_db(tmp_path)
        try:
            assert db_path.is_relative_to(tmp_path)
            memory_id = _insert_legacy_memory(conn, "alpha beta searchable", content_type="fact")
            conn.execute(
                """INSERT INTO file_hashes (file_path, content_hash, indexed_at, chunk_count)
                VALUES ('legacy.md', 'hash-old', '2026-10-01T00:00:00+00:00', 1)"""
            )
            conn.execute(
                "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                (memory_id, db._serialize_f32(mock_embedding("alpha beta searchable"))),
            )
            conn.commit()

            before = dict(
                conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            )
            file_hash_before = dict(
                conn.execute("SELECT * FROM file_hashes WHERE file_path = 'legacy.md'").fetchone()
            )
            fts_row = conn.execute(
                "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'alpha'"
            ).fetchone()
            vec_row = conn.execute(
                "SELECT id FROM memory_vec WHERE id = ?", (memory_id,)
            ).fetchone()
            assert fts_row[0] == memory_id
            assert vec_row[0] == memory_id

            migrate_v3_to_v4(conn)

            after = dict(
                conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            )
            for key in before:
                assert after[key] == before[key]
            file_hash_after = dict(
                conn.execute("SELECT * FROM file_hashes WHERE file_path = 'legacy.md'").fetchone()
            )
            for key in file_hash_before:
                assert file_hash_after[key] == file_hash_before[key]
            fts_row = conn.execute(
                "SELECT rowid FROM memory_fts WHERE memory_fts MATCH 'alpha'"
            ).fetchone()
            vec_row = conn.execute(
                "SELECT id FROM memory_vec WHERE id = ?", (memory_id,)
            ).fetchone()
            assert fts_row[0] == memory_id
            assert vec_row[0] == memory_id
        finally:
            conn.close()


class TestMigrateRollback:
    def test_v2_to_v3_half_failure_leaves_no_version_claim(self, tmp_path, monkeypatch):
        conn, _ = _make_v2_db(tmp_path)
        try:
            bad_sql = "CREATE TABLE schema_version (id INTEGER PRIMARY KEY); SELECT missing"
            monkeypatch.setattr("memory_v3.migration.V3_NEW_TABLES_SQL", bad_sql)

            with pytest.raises(sqlite3.OperationalError):
                migrate_v2_to_v3(conn)

            assert "governance_layer" not in _memory_columns(conn)
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("SELECT version FROM schema_version").fetchone()
        finally:
            conn.close()

    def test_v3_to_v4_failure_rolls_back_columns_and_version_claim(self, tmp_path, monkeypatch):
        conn, _ = _make_v3_db(tmp_path)
        try:
            bad_indexes = V4_INDEXES_SQL + [
                "CREATE INDEX idx_bad_missing_column ON memories(missing_column)"
            ]
            monkeypatch.setattr("memory_v3.migration.V4_INDEXES_SQL", bad_indexes)

            with pytest.raises(sqlite3.OperationalError):
                migrate_v3_to_v4(conn)

            col_names = _memory_columns(conn)
            for col_name, _ in V4_NEW_COLUMNS:
                assert col_name not in col_names

            latest = conn.execute(
                "SELECT version FROM schema_version ORDER BY id DESC LIMIT 1"
            ).fetchone()["version"]
            assert latest == 3
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        finally:
            conn.close()

    def test_malformed_v3_rejected_before_v4_mutation(self, tmp_path):
        conn, _ = _make_v3_db(tmp_path)
        try:
            conn.execute("DROP TRIGGER memories_ai")
            conn.commit()
            before = _schema_snapshot(conn)

            with pytest.raises(RuntimeError, match="not a complete v3 schema"):
                migrate_v3_to_v4(conn)

            assert _schema_snapshot(conn) == before
            assert "owner" not in _memory_columns(conn)
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        finally:
            conn.close()

    def test_existing_v5_like_db_rejected_without_downgrade(self, tmp_path):
        conn, _ = _make_v4_db(tmp_path)
        try:
            conn.execute("ALTER TABLE memories ADD COLUMN source_generation_id TEXT")
            conn.execute("ALTER TABLE memories ADD COLUMN deleted_at TEXT")
            conn.execute("CREATE TABLE source_generations (source_id TEXT)")
            conn.execute(
                "INSERT INTO schema_version (version, migrated_at, notes) "
                "VALUES (5, '2026-01-01T00:00:00+00:00', 'v5-like')"
            )
            conn.execute("PRAGMA user_version = 5")
            conn.commit()
            before = _schema_snapshot(conn)

            with pytest.raises(RuntimeError, match="v5-to-v4 downgrade is not supported"):
                migrate_v3_to_v4(conn)

            assert _schema_snapshot(conn) == before
        finally:
            conn.close()
