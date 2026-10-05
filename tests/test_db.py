"""
Tests for memory_v3.db -- the database layer.

Covers schema creation, CRUD, search, sensory buffer, write queue,
action logging, and v3-specific stats.
"""

import json
import sqlite3
from pathlib import Path

import pytest
import sqlite_vec

from memory_v3 import config as config_module
from memory_v3 import db
from memory_v3.migration import V3_NEW_COLUMNS, migrate_v3_to_v4
from memory_v3.retrieval.filters import RetrievalFilter

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

REMOVED_ACTIVE_DB_IDENTIFIERS = (
    V5_ONLY_MEMORY_COLUMNS
    | (V5_ONLY_FILE_HASH_COLUMNS - {"source_id"})
    | V5_ONLY_TABLES
    | {
        "activate_source_generation",
        "record_generation_failure",
        "get_active_source_generation_payloads",
        "has_unresolved_graph_rebuild_debt",
        "resolve_graph_rebuild_debt",
        "deactivate_source_for_policy_absence",
    }
)
REMOVED_ACTIVE_CONFIG_IDENTIFIERS = {
    "generation_activation_enabled",
    "GENERATION_ACTIVATION_ENABLED",
}


@pytest.fixture
def db_conn(tmp_path):
    db_path = tmp_path / "memory.db"
    conn = db.init_db(str(db_path))
    assert db_path.is_file()
    assert db_path.is_relative_to(tmp_path)
    yield conn
    conn.close()


def _make_v3_shaped_db(db_path):
    from memory_v3.migration import V3_NEW_TABLES_SQL
    from tests.test_migration import V2_SCHEMA_SQL

    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.executescript(V2_SCHEMA_SQL)
    for col_name, col_def in V3_NEW_COLUMNS:
        conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_def}")
    conn.executescript(V3_NEW_TABLES_SQL)
    conn.execute(
        "INSERT INTO schema_version (version, migrated_at, notes) "
        "VALUES (3, '2026-01-01T00:00:00+00:00', 'synthetic v3')"
    )
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()


def _schema_snapshot(conn):
    objects = conn.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    return {
        "objects": [(row[0], row[1], row[2], row[3]) for row in objects],
        "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
        "latest_schema": conn.execute(
            "SELECT version FROM schema_version ORDER BY id DESC LIMIT 1"
        ).fetchone()[0],
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestInitDb:
    """Schema creation and table existence."""

    EXPECTED_TABLES = [
        "memories",
        "memory_fts",
        "graveyard",
        "agent_offsets",
        "file_hashes",
        "conflicts",
        "compaction_receipts",
        # v3 new tables
        "sensory_buffer",
        "cluster_centroids",
        "action_log",
        "consolidation_log",
        "write_queue",
        "schema_version",
        # vec0 virtual table
        "memory_vec",
    ]

    def test_init_db(self, db_conn):
        """init_db should create all v2 + v3 tables."""
        rows = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        table_names = {row["name"] for row in rows}

        for expected in self.EXPECTED_TABLES:
            assert expected in table_names, f"Missing table: {expected}"

    def test_schema_version_recorded(self, db_conn):
        """A fresh init should record the latest schema version."""
        row = db_conn.execute(
            "SELECT version FROM schema_version ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row["version"] == db.SCHEMA_VERSION
        user_version = db_conn.execute("PRAGMA user_version").fetchone()[0]
        assert user_version == db.SCHEMA_VERSION

    def test_fresh_schema_has_exact_v4_metadata_and_no_v5_lifecycle(self, db_conn):
        cols = db_conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = {row["name"] for row in cols}
        expected = {
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
            "source_file",
            "source_hash",
            "source_line",
            "created_at",
            "updated_at",
            "last_verified_at",
            "verification_status",
        }
        assert expected <= col_names
        assert V5_ONLY_MEMORY_COLUMNS.isdisjoint(col_names)

        indexes = db_conn.execute("PRAGMA index_list(memories)").fetchall()
        index_names = {row["name"] for row in indexes}
        assert {
            "idx_memories_owner",
            "idx_memories_namespace",
            "idx_memories_owner_namespace",
            "idx_memories_source_id",
            "idx_memories_chunk_id",
            "idx_memories_expiry_at",
        } <= index_names
        assert {
            "idx_memories_source_generation_id",
            "idx_memories_source_generation_status",
            "idx_memories_deleted_at",
        }.isdisjoint(index_names)

        file_hash_cols = {
            row["name"] for row in db_conn.execute("PRAGMA table_info(file_hashes)").fetchall()
        }
        assert V5_ONLY_FILE_HASH_COLUMNS.isdisjoint(file_hash_cols)

    def test_fresh_schema_has_no_generation_staging_checkpoint_or_debt_tables(self, db_conn):
        tables = {
            row["name"]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        assert V5_ONLY_TABLES.isdisjoint(tables)

    def test_active_db_source_contains_no_removed_v5_identifiers(self):
        db_source = Path(db.__file__).read_text()
        allowed_start = db_source.index("FORBIDDEN_V5_MEMORY_COLUMNS")
        allowed_end = db_source.index("def _serialize_f32")
        executable_source = db_source[:allowed_start] + db_source[allowed_end:]
        assert [
            token for token in REMOVED_ACTIVE_DB_IDENTIFIERS if token in executable_source
        ] == []

    def test_active_config_source_contains_no_generation_activation_switch(self):
        config_source = Path(config_module.__file__).read_text()
        assert [
            token for token in REMOVED_ACTIVE_CONFIG_IDENTIFIERS if token in config_source
        ] == []

    def test_init_db_refuses_existing_v3_shape_without_implicit_migration(self, tmp_path):
        db_path = tmp_path / "legacy-v3.db"
        _make_v3_shaped_db(db_path)
        before = db_path.read_bytes()

        with pytest.raises(RuntimeError, match="run explicit migration first"):
            db.init_db(str(db_path))

        after = db_path.read_bytes()
        assert after == before
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            cols = conn.execute("PRAGMA table_info(memories)").fetchall()
            indexes = conn.execute("PRAGMA index_list(memories)").fetchall()
            col_names = {row[1] for row in cols}
            index_names = {row[1] for row in indexes}

            assert "owner" not in col_names
            assert "project_updated_at" not in col_names
            assert "idx_memories_owner" not in index_names
            assert "idx_memories_expiry_at" not in index_names
            latest = conn.execute(
                "SELECT version FROM schema_version ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
            assert latest == 3
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        finally:
            conn.close()

    def test_explicit_migrate_v3_to_v4_succeeds_and_is_idempotent(self, tmp_path):
        db_path = tmp_path / "legacy-v3.db"
        _make_v3_shaped_db(db_path)
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            first = migrate_v3_to_v4(conn)
            second = migrate_v3_to_v4(conn)
            cols = conn.execute("PRAGMA table_info(memories)").fetchall()
            indexes = conn.execute("PRAGMA index_list(memories)").fetchall()
            col_names = {row[1] for row in cols}
            index_names = {row[1] for row in indexes}

            assert first["columns_added"] == 12
            assert second["columns_added"] == 0
            assert "owner" in col_names
            assert "project_updated_at" in col_names
            assert "idx_memories_owner" in index_names
            assert "idx_memories_expiry_at" in index_names
            latest = conn.execute(
                "SELECT version FROM schema_version ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
            assert latest == 4
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        finally:
            conn.close()

    def test_explicit_v3_to_v4_reopens_normally(self, tmp_path):
        db_path = tmp_path / "legacy-v3.db"
        _make_v3_shaped_db(db_path)
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            migrate_v3_to_v4(conn)
        finally:
            conn.close()

        conn = db.init_db(str(db_path))
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        finally:
            conn.close()

    def test_existing_empty_sqlite_db_is_allowed(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.execute("PRAGMA user_version = 0")
        conn.close()

        conn = db.init_db(str(db_path))
        try:
            table_names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
            }
            assert set(self.EXPECTED_TABLES) <= table_names
            assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
        finally:
            conn.close()

    def test_unrelated_existing_schema_without_memories_is_refused_unchanged(self, tmp_path):
        db_path = tmp_path / "partial.db"
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        before = db_path.read_bytes()

        with pytest.raises(RuntimeError, match="run explicit migration first"):
            db.init_db(str(db_path))

        assert db_path.read_bytes() == before
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            objects = conn.execute(
                "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            assert [row[0] for row in objects] == ["unrelated"]
        finally:
            conn.close()

    @pytest.mark.parametrize(
        "mutation",
        [
            "DROP TABLE conflicts",
            "DROP TRIGGER memories_ai",
            "DROP INDEX idx_memories_owner",
            "PRAGMA user_version = 3",
            "INSERT INTO schema_version (version, migrated_at, notes) "
            "VALUES (3, '2026-01-01T00:00:00+00:00', 'bad downgrade')",
        ],
    )
    def test_incomplete_v4_is_refused_unchanged(self, tmp_path, mutation):
        db_path = tmp_path / "incomplete-v4.db"
        conn = db.init_db(str(db_path))
        conn.close()
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.execute(mutation)
        conn.commit()
        conn.close()
        before = db_path.read_bytes()

        with pytest.raises(RuntimeError, match="run explicit migration first"):
            db.init_db(str(db_path))

        assert db_path.read_bytes() == before

    def test_complete_v4_reopens_without_schema_drift(self, tmp_path):
        db_path = tmp_path / "complete-v4.db"
        conn = db.init_db(str(db_path))
        before = _schema_snapshot(conn)
        conn.close()

        conn = db.init_db(str(db_path))
        try:
            after = _schema_snapshot(conn)
        finally:
            conn.close()

        assert after == before

    def test_existing_v5_like_db_is_rejected_unchanged(self, tmp_path):
        db_path = tmp_path / "v5-like.db"
        conn = db.init_db(str(db_path))
        conn.execute("ALTER TABLE memories ADD COLUMN source_generation_id TEXT")
        conn.execute("ALTER TABLE memories ADD COLUMN deleted_at TEXT")
        conn.execute("ALTER TABLE file_hashes ADD COLUMN active_generation_id TEXT")
        conn.execute("CREATE TABLE source_generations (source_id TEXT)")
        conn.execute(
            "INSERT INTO schema_version (version, migrated_at, notes) "
            "VALUES (5, '2026-01-01T00:00:00+00:00', 'v5-like')"
        )
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
        before = _schema_snapshot(conn)
        conn.close()
        before_bytes = db_path.read_bytes()

        with pytest.raises(RuntimeError, match="run explicit migration first"):
            db.init_db(str(db_path))

        assert db_path.read_bytes() == before_bytes
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            after = _schema_snapshot(conn)
        finally:
            conn.close()
        assert after == before

    def test_v4_versioned_db_with_forbidden_v5_extras_is_rejected_unchanged(self, tmp_path):
        db_path = tmp_path / "v4-with-forbidden-extras.db"
        conn = db.init_db(str(db_path))
        for column in db.FORBIDDEN_V5_MEMORY_COLUMNS:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {column} TEXT")
        for column in db.FORBIDDEN_V5_FILE_HASH_COLUMNS:
            conn.execute(f"ALTER TABLE file_hashes ADD COLUMN {column} TEXT")
        for name in sorted(db.FORBIDDEN_V5_SCHEMA_OBJECTS):
            if name == "indexing_debt":
                conn.execute(f"CREATE VIEW {name} AS SELECT 1 AS id")
            else:
                conn.execute(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY)")
        conn.commit()
        before = _schema_snapshot(conn)
        assert before["user_version"] == db.SCHEMA_VERSION
        assert before["latest_schema"] == db.SCHEMA_VERSION
        conn.close()
        before_bytes = db_path.read_bytes()

        with pytest.raises(RuntimeError, match="run explicit migration first"):
            db.init_db(str(db_path))

        assert db_path.read_bytes() == before_bytes
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            after = _schema_snapshot(conn)
        finally:
            conn.close()
        assert after == before


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestAddAndGetMemory:
    """Round-trip: add then get, including v3 fields."""

    def test_add_and_get_memory(self, db_conn, mock_embedding):
        emb = mock_embedding("test memory content")
        mid = db.add_memory(
            db_conn,
            content="test memory content",
            embedding=emb,
            content_type="fact",
            governance_layer=2,
            structure_type="atomic",
            surprise_score=0.8,
        )
        assert mid > 0

        mem = db.get_memory(db_conn, mid)
        assert mem is not None
        assert mem["content"] == "test memory content"
        assert mem["governance_layer"] == 2
        assert mem["structure_type"] == "atomic"
        assert abs(mem["surprise_score"] - 0.8) < 0.01

    def test_get_nonexistent_memory(self, db_conn):
        assert db.get_memory(db_conn, 99999) is None

    def test_add_protected_auto_governance(self, db_conn, mock_embedding):
        """Protected memories auto-classify to governance_layer 1."""
        emb = mock_embedding("identity core")
        mid = db.add_memory(
            db_conn,
            content="identity core",
            embedding=emb,
            protected=True,
        )
        mem = db.get_memory(db_conn, mid)
        assert mem["governance_layer"] == 1
        assert mem["protected"] == 1

    def test_metadata_roundtrip(self, db_conn, mock_embedding):
        mid = db.add_memory(
            db_conn,
            content="derived project chunk",
            embedding=mock_embedding("derived project chunk"),
            source_file="source.md",
            source_line=42,
            source_hash="hash-a",
            owner="base",
            namespace="project-x",
            sensitivity="internal",
            record_kind="derived",
            source_id="source-1",
            chunk_id="chunk-1",
            chunk_index=2,
            chunk_count=5,
            source_updated_at="2026-10-01T00:00:00+00:00",
            expiry_at="2026-12-01T00:00:00+00:00",
            project_status="active",
            project_updated_at="2026-10-02T00:00:00+00:00",
        )

        mem = db.get_memory(db_conn, mid)
        assert mem["source_file"] == "source.md"
        assert mem["source_line"] == 42
        assert mem["source_hash"] == "hash-a"
        assert mem["owner"] == "base"
        assert mem["namespace"] == "project-x"
        assert mem["sensitivity"] == "internal"
        assert mem["record_kind"] == "derived"
        assert mem["source_id"] == "source-1"
        assert mem["chunk_id"] == "chunk-1"
        assert mem["chunk_index"] == 2
        assert mem["chunk_count"] == 5
        assert mem["source_updated_at"] == "2026-10-01T00:00:00+00:00"
        assert mem["expiry_at"] == "2026-12-01T00:00:00+00:00"
        assert mem["project_status"] == "active"
        assert mem["project_updated_at"] == "2026-10-02T00:00:00+00:00"

    def test_add_rejects_invalid_record_kind(self, db_conn, mock_embedding):
        with pytest.raises(ValueError, match="derived, handoff"):
            db.add_memory(
                db_conn,
                content="bad kind",
                embedding=mock_embedding("bad kind"),
                record_kind="raw",
            )

    def test_add_rejects_durable_record_kind(self, db_conn, mock_embedding):
        with pytest.raises(ValueError, match="derived, handoff"):
            db.add_memory(
                db_conn,
                content="durable kind",
                embedding=mock_embedding("durable kind"),
                record_kind="durable",
            )


class TestUpdateMemory:
    """update_memory with v3 fields."""

    def test_update_memory_governance(self, db_conn, mock_embedding):
        emb = mock_embedding("some fact")
        mid = db.add_memory(db_conn, content="some fact", embedding=emb)

        success = db.update_memory(db_conn, mid, governance_layer=1)
        assert success is True

        mem = db.get_memory(db_conn, mid)
        assert mem["governance_layer"] == 1

    def test_update_verification_status(self, db_conn, mock_embedding):
        emb = mock_embedding("verify me")
        mid = db.add_memory(db_conn, content="verify me", embedding=emb)

        db.update_memory(db_conn, mid, verification_status="verified")
        mem = db.get_memory(db_conn, mid)
        assert mem["verification_status"] == "verified"
        assert mem["last_verified_at"] is not None

    def test_update_no_fields_returns_false(self, db_conn, mock_embedding):
        emb = mock_embedding("noop")
        mid = db.add_memory(db_conn, content="noop", embedding=emb)
        assert db.update_memory(db_conn, mid) is False

    def test_update_metadata_fields(self, db_conn, mock_embedding):
        mid = db.add_memory(db_conn, content="handoff", embedding=mock_embedding("handoff"))

        assert (
            db.update_memory(
                db_conn,
                mid,
                owner="base",
                namespace="ops",
                record_kind="handoff",
                source_id="src-2",
                chunk_id="chunk-2",
                chunk_index=0,
                chunk_count=1,
                expiry_at="2027-01-01T00:00:00+00:00",
                project_status="paused",
            )
            is True
        )

        mem = db.get_memory(db_conn, mid)
        assert mem["owner"] == "base"
        assert mem["namespace"] == "ops"
        assert mem["record_kind"] == "handoff"
        assert mem["source_id"] == "src-2"
        assert mem["chunk_id"] == "chunk-2"
        assert mem["chunk_index"] == 0
        assert mem["chunk_count"] == 1
        assert mem["expiry_at"] == "2027-01-01T00:00:00+00:00"
        assert mem["project_status"] == "paused"

    def test_update_rejects_invalid_record_kind(self, db_conn, mock_embedding):
        mid = db.add_memory(db_conn, content="bad update", embedding=mock_embedding("bad update"))
        with pytest.raises(ValueError, match="derived, handoff"):
            db.update_memory(db_conn, mid, record_kind="raw")

    def test_update_rejects_durable_record_kind(self, db_conn, mock_embedding):
        mid = db.add_memory(
            db_conn,
            content="durable update",
            embedding=mock_embedding("durable update"),
        )
        with pytest.raises(ValueError, match="derived, handoff"):
            db.update_memory(db_conn, mid, record_kind="durable")


# ---------------------------------------------------------------------------
# Archival
# ---------------------------------------------------------------------------


class TestArchiveMemory:
    """archive_memory respects governance and protected rules."""

    def test_archive_memory(self, db_conn, mock_embedding):
        """A normal memory (layer 3) can be archived."""
        emb = mock_embedding("ephemeral note")
        mid = db.add_memory(
            db_conn,
            content="ephemeral note",
            embedding=emb,
            governance_layer=4,
        )
        result = db.archive_memory(db_conn, mid, reason="test")
        assert result is True

        # Should not be retrievable
        assert db.get_memory(db_conn, mid) is None

        # Should be in graveyard
        grave = db_conn.execute("SELECT * FROM graveyard WHERE memory_id = ?", (mid,)).fetchone()
        assert grave is not None

    def test_archive_protected_blocked(self, db_conn, mock_embedding):
        """Protected memories should NOT be archived."""
        emb = mock_embedding("protected")
        mid = db.add_memory(
            db_conn,
            content="protected",
            embedding=emb,
            protected=True,
        )
        result = db.archive_memory(db_conn, mid, reason="test")
        assert result is False

    def test_archive_constitutional(self, db_conn, mock_embedding):
        """
        governance_layer=1 (constitutional) should block archival at the DB level.
        The archive function checks governance_layer in (1, 2) and refuses.
        """
        emb = mock_embedding("constitutional mem")
        mid = db.add_memory(
            db_conn,
            content="constitutional mem",
            embedding=emb,
            governance_layer=1,
        )
        # Force protected=0 so we test governance_layer check specifically
        db_conn.execute("UPDATE memories SET protected = 0 WHERE id = ?", (mid,))
        db_conn.commit()

        result = db.archive_memory(db_conn, mid, reason="test")
        assert result is False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """hybrid_search with BM25 + vector fusion."""

    OWNER = "test-owner"

    def test_hybrid_search(self, db_conn, mock_embedding):
        """Add 3 memories, search should return ranked results."""
        contents = [
            "Python is a programming language",
            "SQLite is an embedded database",
            "Python and SQLite work well together",
        ]
        for c in contents:
            db.add_memory(db_conn, content=c, embedding=mock_embedding(c), owner=self.OWNER)

        query = "Python programming"
        results = db.hybrid_search(
            db_conn,
            query_text=query,
            query_embedding=mock_embedding(query),
            limit=10,
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )
        assert len(results) > 0
        # Results should have hybrid_score
        assert "hybrid_score" in results[0]

    def test_hybrid_search_empty_db(self, db_conn, mock_embedding):
        results = db.hybrid_search(
            db_conn,
            query_text="anything",
            query_embedding=mock_embedding("anything"),
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )
        assert results == []

    def test_ownerless_hybrid_search_raises_before_sql(self, mock_embedding):
        class NoSqlConnection:
            calls = 0

            def execute(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("SQL must not run")

        conn = NoSqlConnection()

        with pytest.raises(ValueError, match="owner"):
            db.hybrid_search(conn, "needle", mock_embedding("needle"), retrieval_filter=None)
        with pytest.raises(ValueError, match="owner"):
            db.hybrid_search(
                conn,
                "needle",
                mock_embedding("needle"),
                retrieval_filter=RetrievalFilter(),
            )

        assert conn.calls == 0

    def test_hybrid_search_parameterizes_content_type_filter(self, db_conn, mock_embedding):
        db.add_memory(
            db_conn,
            content="needle fact",
            embedding=mock_embedding("needle fact"),
            content_type="fact",
            owner=self.OWNER,
        )

        results = db.hybrid_search(
            db_conn,
            query_text="needle",
            query_embedding=mock_embedding("needle"),
            content_type="fact' OR 1=1 --",
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )
        assert results == []

    def test_hybrid_search_normal_filters(self, db_conn, mock_embedding):
        fact_id = db.add_memory(
            db_conn,
            content="shared filter term fact",
            embedding=mock_embedding("shared filter term fact"),
            content_type="fact",
            governance_layer=2,
            owner=self.OWNER,
        )
        episode_id = db.add_memory(
            db_conn,
            content="shared filter term episode",
            embedding=mock_embedding("shared filter term episode"),
            content_type="episode",
            governance_layer=4,
            owner=self.OWNER,
        )

        fact_results = db.hybrid_search(
            db_conn,
            query_text="shared filter term",
            query_embedding=mock_embedding("shared filter term"),
            content_type="fact",
            governance_layer=2,
            limit=10,
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )
        episode_results = db.hybrid_search(
            db_conn,
            query_text="shared filter term",
            query_embedding=mock_embedding("shared filter term"),
            governance_layer=4,
            limit=10,
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )

        assert {row["id"] for row in fact_results} == {fact_id}
        assert {row["id"] for row in episode_results} == {episode_id}

    def test_hybrid_search_excludes_cross_owner_rows(self, db_conn, mock_embedding):
        owner_id = db.add_memory(
            db_conn,
            content="owned shared needle",
            embedding=mock_embedding("owned shared needle"),
            owner=self.OWNER,
        )
        other_id = db.add_memory(
            db_conn,
            content="other shared needle",
            embedding=mock_embedding("other shared needle"),
            owner="other-owner",
        )

        results = db.hybrid_search(
            db_conn,
            query_text="shared needle",
            query_embedding=mock_embedding("shared needle"),
            limit=10,
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )

        assert {row["id"] for row in results} == {owner_id}
        assert other_id not in {row["id"] for row in results}


class TestKeywordSearch:
    """Pure BM25 keyword search via FTS5."""

    OWNER = "test-owner"

    def test_keyword_search(self, db_conn, mock_embedding):
        db.add_memory(
            db_conn,
            content="Rust memory safety",
            embedding=mock_embedding("Rust memory safety"),
            owner=self.OWNER,
        )
        db.add_memory(
            db_conn,
            content="Go concurrency model",
            embedding=mock_embedding("Go concurrency model"),
            owner=self.OWNER,
        )

        results = db.keyword_search(
            db_conn,
            keywords="Rust",
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )
        assert len(results) >= 1
        assert "Rust" in results[0]["content"]

    def test_keyword_search_no_match(self, db_conn, mock_embedding):
        db.add_memory(
            db_conn,
            content="hello world",
            embedding=mock_embedding("hello world"),
            owner=self.OWNER,
        )
        results = db.keyword_search(
            db_conn,
            keywords="zzznonexistent",
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )
        assert results == []

    def test_ownerless_keyword_search_raises_before_sql(self):
        class NoSqlConnection:
            calls = 0

            def execute(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("SQL must not run")

        conn = NoSqlConnection()

        with pytest.raises(ValueError, match="owner"):
            db.keyword_search(conn, "needle", retrieval_filter=None)
        with pytest.raises(ValueError, match="owner"):
            db.keyword_search(conn, "needle", retrieval_filter=RetrievalFilter())

        assert conn.calls == 0

    def test_keyword_search_excludes_cross_owner_rows(self, db_conn, mock_embedding):
        owner_id = db.add_memory(
            db_conn,
            content="owned keyword needle",
            embedding=mock_embedding("owned keyword needle"),
            owner=self.OWNER,
        )
        other_id = db.add_memory(
            db_conn,
            content="other keyword needle",
            embedding=mock_embedding("other keyword needle"),
            owner="other-owner",
        )

        results = db.keyword_search(
            db_conn,
            keywords="keyword needle",
            retrieval_filter=RetrievalFilter(owner=self.OWNER),
        )

        assert {row["id"] for row in results} == {owner_id}
        assert other_id not in {row["id"] for row in results}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestGetStats:
    """get_stats includes v3 governance_layer_counts."""

    def test_get_stats_v3(self, db_conn, mock_embedding):
        # Add memories in different governance layers
        db.add_memory(
            db_conn,
            content="identity fact",
            embedding=mock_embedding("identity fact"),
            governance_layer=1,
        )
        db.add_memory(
            db_conn,
            content="a decision",
            embedding=mock_embedding("a decision"),
            governance_layer=2,
        )
        db.add_memory(
            db_conn,
            content="plain fact",
            embedding=mock_embedding("plain fact"),
            governance_layer=3,
        )
        db.add_memory(
            db_conn,
            content="an episode",
            embedding=mock_embedding("an episode"),
            governance_layer=4,
        )

        stats = db.get_stats(db_conn)

        assert stats["total_active"] == 4
        assert "governance_layer_counts" in stats
        assert stats["governance_layer_counts"][1] == 1
        assert stats["governance_layer_counts"][2] == 1
        assert stats["governance_layer_counts"][3] == 1
        assert stats["governance_layer_counts"][4] == 1

    def test_stats_sensory_and_write_queue(self, db_conn, mock_embedding):
        """Stats should report pending sensory items and write queue items."""
        db.add_to_sensory_buffer(db_conn, content="buf item", content_hash="abc123")
        db.enqueue_write(db_conn, operation="add", payload={"content": "test"})

        stats = db.get_stats(db_conn)
        assert stats["sensory_buffer_pending"] == 1
        assert stats["write_queue_pending"] == 1


# ---------------------------------------------------------------------------
# Sensory Buffer
# ---------------------------------------------------------------------------


class TestSensoryBuffer:
    """add_to_sensory_buffer, get_pending_sensory, mark_sensory_processed."""

    def test_add_to_sensory_buffer(self, db_conn):
        buf_id = db.add_to_sensory_buffer(
            db_conn,
            content="new observation",
            content_hash="hash123",
            content_type="episode",
            source="conversation",
            novelty_score=0.9,
        )
        assert buf_id > 0

    def test_get_pending_sensory(self, db_conn):
        db.add_to_sensory_buffer(db_conn, content="item 1", content_hash="h1")
        db.add_to_sensory_buffer(db_conn, content="item 2", content_hash="h2")

        pending = db.get_pending_sensory(db_conn)
        assert len(pending) == 2
        assert pending[0]["content"] == "item 1"

    def test_mark_sensory_processed(self, db_conn):
        buf_id = db.add_to_sensory_buffer(db_conn, content="process me", content_hash="hp")

        db.mark_sensory_processed(
            db_conn,
            buffer_id=buf_id,
            processed=1,
            decision="ADD",
            decision_reason="novel content",
        )

        # Should no longer appear in pending
        pending = db.get_pending_sensory(db_conn)
        assert len(pending) == 0

        # Verify the record was updated
        row = db_conn.execute("SELECT * FROM sensory_buffer WHERE id = ?", (buf_id,)).fetchone()
        assert row["processed"] == 1
        assert row["decision"] == "ADD"


# ---------------------------------------------------------------------------
# Write Queue
# ---------------------------------------------------------------------------


class TestWriteQueue:
    """enqueue_write, get_pending_writes, mark_write_complete."""

    def test_enqueue_write(self, db_conn):
        wid = db.enqueue_write(
            db_conn,
            operation="add_memory",
            payload={"content": "queued memory"},
            priority=3,
        )
        assert wid > 0

    def test_get_pending_writes(self, db_conn):
        db.enqueue_write(db_conn, operation="add", payload={"a": 1}, priority=5)
        db.enqueue_write(db_conn, operation="update", payload={"b": 2}, priority=1)

        pending = db.get_pending_writes(db_conn)
        assert len(pending) == 2
        # Lower priority number = higher priority = comes first
        assert pending[0]["priority"] == 1

    def test_mark_write_complete(self, db_conn):
        wid = db.enqueue_write(db_conn, operation="add", payload="test")

        db.mark_write_complete(db_conn, wid, status="completed")

        pending = db.get_pending_writes(db_conn)
        assert len(pending) == 0

        row = db_conn.execute("SELECT * FROM write_queue WHERE id = ?", (wid,)).fetchone()
        assert row["status"] == "completed"
        assert row["completed_at"] is not None

    def test_mark_write_failed(self, db_conn):
        wid = db.enqueue_write(db_conn, operation="add", payload="fail test")

        db.mark_write_complete(db_conn, wid, status="failed", error="disk full")

        row = db_conn.execute("SELECT * FROM write_queue WHERE id = ?", (wid,)).fetchone()
        assert row["status"] == "failed"
        assert row["error"] == "disk full"


# ---------------------------------------------------------------------------
# Action Log
# ---------------------------------------------------------------------------


class TestActionLog:
    """log_action round-trip."""

    def test_log_action(self, db_conn):
        log_id = db.log_action(
            db_conn,
            new_fact="Sean prefers Python",
            similar_ids=[1, 2, 3],
            action="ADD",
            confidence=0.95,
            model_version="v3-test",
        )
        assert log_id > 0

        row = db_conn.execute("SELECT * FROM action_log WHERE id = ?", (log_id,)).fetchone()
        assert row["new_fact"] == "Sean prefers Python"
        assert row["action"] == "ADD"
        assert json.loads(row["similar_ids"]) == [1, 2, 3]
        assert abs(row["confidence"] - 0.95) < 0.01
        assert row["model_version"] == "v3-test"
