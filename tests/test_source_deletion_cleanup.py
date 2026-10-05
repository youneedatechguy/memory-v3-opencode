import hashlib
import json
from pathlib import Path

import pytest

from memory_v3.config import reset_config
from memory_v3.retrieval.filters import RetrievalFilter


class CountingEmbedder:
    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(768)]


@pytest.fixture(autouse=True)
def server_state(monkeypatch, tmp_path):
    import memory_v3.server as server

    reset_config()
    if server._conn is not None:
        server._conn.close()
    server._conn = None
    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-a")
    yield
    if server._conn is not None:
        server._conn.close()
    server._conn = None
    reset_config()


def configure(monkeypatch, tmp_path, vault: Path) -> None:
    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_VAULT", str(vault))
    monkeypatch.setenv("MEMORY_V3_INDEXING_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-a")
    reset_config()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_reindex(monkeypatch) -> dict:
    import memory_v3.server as server

    embedder = CountingEmbedder()
    monkeypatch.setattr(server, "_get_embedder", lambda: embedder)
    return json.loads(server.reindex())


def count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def source_state(conn, path: str):
    rows = conn.execute(
        "SELECT id, source_id FROM memories WHERE source_file = ? ORDER BY id",
        (path,),
    ).fetchall()
    return [row["id"] for row in rows], (rows[0]["source_id"] if rows else None)


def assert_no_removed_columns_or_tables(conn):
    memory_columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    file_hash_columns = {row[1] for row in conn.execute("PRAGMA table_info(file_hashes)")}
    objects = {row[0] for row in conn.execute("SELECT name FROM sqlite_master").fetchall()}
    assert {
        "source_generation_id",
        "source_generation_status",
        "line_start",
        "line_end",
        "heading_path",
        "rank_weight",
        "rank_marker",
        "historical",
        "deleted_at",
    }.isdisjoint(memory_columns)
    assert {"source_id", "active_generation_id", "deleted_at"}.isdisjoint(file_hash_columns)
    assert {
        "source_generations",
        "staged_source_chunks",
        "indexing_checkpoints",
        "indexing_debt",
    }.isdisjoint(objects)


def test_physical_deletion_cleans_memory_vector_fts_and_hash(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    source = vault / "docs/delete.md"
    write(source, "old delete token searchable\n")
    configure(monkeypatch, tmp_path, vault)
    assert run_reindex(monkeypatch)["errors"] == 0
    deleted_ids, source_id = source_state(server._conn, "docs/delete.md")
    assert deleted_ids

    source.unlink()
    result = run_reindex(monkeypatch)
    second = run_reindex(monkeypatch)

    assert result["deleted_sources"] == 1
    assert result["deleted_memories"] == len(deleted_ids)
    assert "deletion_errors" not in result
    assert second.get("deleted_sources", 0) == 0
    assert count(server._conn, "memories") == 0
    assert count(server._conn, "memory_vec") == 0
    assert count(server._conn, "memory_fts") == 0
    assert (
        server._conn.execute(
            "SELECT 1 FROM file_hashes WHERE file_path = ?",
            ("docs/delete.md",),
        ).fetchone()
        is None
    )
    assert (
        memory_db.keyword_search(
            server._conn,
            "delete",
            retrieval_filter=RetrievalFilter(owner="owner-a"),
        )
        == []
    )
    assert (
        server._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE source_id = ?",
            (source_id,),
        ).fetchone()[0]
        == 0
    )
    assert_no_removed_columns_or_tables(server._conn)


def test_policy_ineligible_source_is_removed_without_generation_or_debt(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    project = vault / "projects/done/CURRENT.md"
    write(project, "Status: active\nLast updated: 2026-10-01\nterminal policy token\n")
    configure(monkeypatch, tmp_path, vault)
    assert run_reindex(monkeypatch)["errors"] == 0

    write(project, "Status: completed\nLast updated: 2026-10-01\nterminal policy token\n")
    result = run_reindex(monkeypatch)

    assert result["deleted_sources"] == 1
    assert result["deleted_memories"] == 1
    assert count(server._conn, "memories") == 0
    assert count(server._conn, "memory_vec") == 0
    assert count(server._conn, "memory_fts") == 0
    assert (
        memory_db.keyword_search(
            server._conn,
            "terminal",
            retrieval_filter=RetrievalFilter(owner="owner-a"),
        )
        == []
    )
    assert_no_removed_columns_or_tables(server._conn)


def test_cross_owner_rows_sharing_path_survive_and_keep_shared_hash(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    source = vault / "docs/shared.md"
    write(source, "owner a shared token\n")
    configure(monkeypatch, tmp_path, vault)
    assert run_reindex(monkeypatch)["errors"] == 0
    memory_db.add_memory(
        server._conn,
        "owner b shared token",
        CountingEmbedder().embed("owner b shared token"),
        source_file="docs/shared.md",
        source_hash="owner-b-hash",
        owner="owner-b",
        namespace="docs",
        record_kind="derived",
        source_id="src:owner-b-shared",
        chunk_id="chk:owner-b-shared",
        chunk_index=0,
        chunk_count=1,
    )

    source.unlink()
    result = run_reindex(monkeypatch)

    assert result["deleted_sources"] == 1
    rows = server._conn.execute("SELECT owner, content FROM memories ORDER BY owner").fetchall()
    assert [(row["owner"], row["content"]) for row in rows] == [("owner-b", "owner b shared token")]
    assert count(server._conn, "memory_vec") == 1
    assert (
        server._conn.execute(
            "SELECT content_hash FROM file_hashes WHERE file_path = ?",
            ("docs/shared.md",),
        ).fetchone()
        is not None
    )
    assert (
        memory_db.keyword_search(
            server._conn,
            "shared",
            retrieval_filter=RetrievalFilter(owner="owner-a"),
        )
        == []
    )
    assert [
        row["content"]
        for row in memory_db.keyword_search(
            server._conn,
            "shared",
            retrieval_filter=RetrievalFilter(owner="owner-b"),
        )
    ] == ["owner b shared token"]


def test_cleanup_failure_rolls_back_everything_and_sanitizes_error(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    source = vault / "docs/delete.md"
    write(source, "rollback deleted token\n")
    configure(monkeypatch, tmp_path, vault)
    assert run_reindex(monkeypatch)["errors"] == 0
    deleted_ids, source_id = source_state(server._conn, "docs/delete.md")

    def fail_after_mutation(conn, cleanup_source_id, memory_ids):
        raise RuntimeError("cleanup boom SECRET_CONTENT")

    monkeypatch.setattr(memory_db.cleanup_source, "test_hook", fail_after_mutation, raising=False)
    source.unlink()
    result = run_reindex(monkeypatch)
    monkeypatch.delattr(memory_db.cleanup_source, "test_hook", raising=False)

    assert json.dumps(result).find("SECRET_CONTENT") == -1
    assert result.get("deleted_sources", 0) == 0
    assert result["deletion_errors"] == [
        {
            "source_path": "docs/delete.md",
            "source_id": source_id,
            "exception_class": "RuntimeError",
        }
    ]
    assert server._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE source_id = ?",
        (source_id,),
    ).fetchone()[0] == len(deleted_ids)
    placeholders = ", ".join("?" for _ in deleted_ids)
    assert server._conn.execute(
        f"SELECT COUNT(*) FROM memory_vec WHERE id IN ({placeholders})",
        deleted_ids,
    ).fetchone()[0] == len(deleted_ids)
    assert memory_db.keyword_search(
        server._conn,
        "deleted",
        retrieval_filter=RetrievalFilter(owner="owner-a"),
    )
    assert (
        server._conn.execute(
            "SELECT 1 FROM file_hashes WHERE file_path = ?",
            ("docs/delete.md",),
        ).fetchone()
        is not None
    )


def test_cleanup_source_does_not_delete_handoffs(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    source = vault / "docs/delete.md"
    write(source, "derived deletion token\n")
    configure(monkeypatch, tmp_path, vault)
    assert run_reindex(monkeypatch)["errors"] == 0
    memory_db.add_memory(
        server._conn,
        "handoff deletion token",
        CountingEmbedder().embed("handoff deletion token"),
        source_file="docs/delete.md",
        owner="owner-a",
        namespace="handoff",
        record_kind="handoff",
        source_id="src:handoff",
        chunk_id="chk:handoff",
        expiry_at="2999-01-01T00:00:00+00:00",
    )

    source.unlink()
    result = run_reindex(monkeypatch)

    assert result["deleted_sources"] == 1
    rows = server._conn.execute("SELECT record_kind, content FROM memories").fetchall()
    assert [(row["record_kind"], row["content"]) for row in rows] == [
        ("handoff", "handoff deletion token")
    ]
    assert count(server._conn, "memory_vec") == 1
