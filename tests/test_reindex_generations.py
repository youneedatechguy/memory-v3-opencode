import hashlib
import json
from pathlib import Path

import pytest

from memory_v3.config import reset_config
from memory_v3.retrieval.filters import RetrievalFilter

TEST_EMBED_MODEL = "test-embed-model"
TEST_EMBED_PROVIDER = "test-provider"
TEST_EMBED_DIM = 768
TEST_OWNER = "owner-a"


class CountingEmbedder:
    def __init__(self, *, fail_on_call: int | None = None, bad: str | None = None):
        self.inputs: list[str] = []
        self.fail_on_call = fail_on_call
        self.bad = bad

    def embed(self, text: str) -> list[float]:
        if self.fail_on_call is not None and len(self.inputs) + 1 == self.fail_on_call:
            raise RuntimeError("embedding boom SECRET_TEXT")
        self.inputs.append(text)
        if self.bad == "length":
            return [0.0]
        if self.bad == "nan":
            return [float("nan")] * TEST_EMBED_DIM
        digest = hashlib.sha256(text.encode()).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(TEST_EMBED_DIM)]


@pytest.fixture(autouse=True)
def server_state(monkeypatch, tmp_path):
    import memory_v3.server as server

    reset_config()
    if server._conn is not None:
        server._conn.close()
    server._conn = None
    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-a")
    yield
    if server._conn is not None:
        server._conn.close()
    server._conn = None
    reset_config()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def configure(monkeypatch, tmp_path, vault: Path) -> None:
    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_VAULT", str(vault))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", TEST_EMBED_MODEL)
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", TEST_EMBED_PROVIDER)
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", str(TEST_EMBED_DIM))
    monkeypatch.setenv("MEMORY_V3_INDEXING_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-a")
    reset_config()


def long_note(marker: str = "FINAL_QUARTER_RETRIEVAL_MARKER") -> str:
    lines = ["# Long\n", "\n", "## First\n"]
    lines += [f"line {index:03d} {'x' * 90}\n" for index in range(90)]
    lines[75] = marker + "\n"
    return "".join(lines)


def counts(conn) -> dict[str, int]:
    return {
        "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
        "vectors": conn.execute("SELECT COUNT(*) FROM memory_vec").fetchone()[0],
        "fts": conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0],
    }


def source_snapshot(conn, path: str) -> dict:
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT id, content, source_hash, source_id, chunk_id, chunk_index, chunk_count
            FROM memories WHERE source_file = ? ORDER BY chunk_index, id
            """,
            (path,),
        ).fetchall()
    ]
    marker = conn.execute(
        "SELECT content_hash, chunk_count FROM file_hashes WHERE file_path = ?",
        (path,),
    ).fetchone()
    return {
        "rows": rows,
        "hash": None if marker is None else marker["content_hash"],
        "chunk_count": None if marker is None else marker["chunk_count"],
        "counts": counts(conn),
    }


def test_successful_multichunk_reindex_retrieves_final_quarter_and_context(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    write(vault / "research/long.md", long_note())
    configure(monkeypatch, tmp_path, vault)
    embedder = CountingEmbedder()
    monkeypatch.setattr(server, "get_embedder", lambda: embedder)

    result = json.loads(server.reindex())
    keyword = memory_db.keyword_search(
        server._conn,
        "FINAL_QUARTER_RETRIEVAL_MARKER",
        retrieval_filter=RetrievalFilter(owner=TEST_OWNER),
    )

    assert result == {"indexed": 1, "skipped": 0, "errors": 0, "force": False}
    assert len(keyword) == 1
    assert keyword[0]["chunk_count"] > 1
    assert keyword[0]["chunk_index"] == keyword[0]["chunk_count"] - 1
    assert "Source: research/long.md" in keyword[0]["content"]
    assert "Lines:" in keyword[0]["content"]
    assert "Headings: Long > First" in keyword[0]["content"]


def test_changed_source_replaces_old_memory_vector_fts_and_hash(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    path = vault / "docs/note.md"
    write(path, "# Note\n\noldtoken content\n")
    configure(monkeypatch, tmp_path, vault)
    monkeypatch.setattr(server, "get_embedder", lambda: CountingEmbedder())
    assert json.loads(server.reindex())["errors"] == 0
    before = source_snapshot(server._conn, "docs/note.md")

    write(path, "# Note\n\nnewtoken content\n")
    assert json.loads(server.reindex())["errors"] == 0
    after = source_snapshot(server._conn, "docs/note.md")

    assert after["counts"] == before["counts"]
    assert after["hash"] == hashlib.sha256(b"# Note\n\nnewtoken content\n").hexdigest()
    assert after["rows"][0]["id"] != before["rows"][0]["id"]
    assert memory_db.keyword_search(
        server._conn, "newtoken", retrieval_filter=RetrievalFilter(owner=TEST_OWNER)
    )
    assert (
        memory_db.keyword_search(
            server._conn, "oldtoken", retrieval_filter=RetrievalFilter(owner=TEST_OWNER)
        )
        == []
    )


def test_embedding_failure_and_malformed_vectors_preserve_old_state_and_hash(monkeypatch, tmp_path):
    import memory_v3.server as server

    vault = tmp_path / "vault"
    path = vault / "docs/note.md"
    write(path, "old stable content\n")
    configure(monkeypatch, tmp_path, vault)
    monkeypatch.setattr(server, "get_embedder", lambda: CountingEmbedder())
    assert json.loads(server.reindex())["errors"] == 0
    before = source_snapshot(server._conn, "docs/note.md")

    write(path, "new unstable content\n")
    for bad_embedder in (
        CountingEmbedder(fail_on_call=1),
        CountingEmbedder(bad="length"),
        CountingEmbedder(bad="nan"),
    ):
        monkeypatch.setattr(server, "get_embedder", lambda embedder=bad_embedder: embedder)
        raw = server.reindex()
        result = json.loads(raw)
        assert result["errors"] == 1
        assert "SECRET_TEXT" not in raw
        assert source_snapshot(server._conn, "docs/note.md") == before


@pytest.mark.parametrize("failure_phase", ["after_delete", "after_partial_insert"])
def test_injected_transaction_failure_rolls_back_all_state(monkeypatch, tmp_path, failure_phase):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    path = vault / "docs/note.md"
    write(path, "old rollback content\n")
    configure(monkeypatch, tmp_path, vault)
    monkeypatch.setattr(server, "get_embedder", lambda: CountingEmbedder())
    assert json.loads(server.reindex())["errors"] == 0
    before = source_snapshot(server._conn, "docs/note.md")

    def fail_hook(conn, phase):
        if phase == failure_phase:
            raise RuntimeError("transaction boom SECRET")

    monkeypatch.setattr(memory_db.replace_source_chunks, "test_hook", fail_hook, raising=False)
    write(path, "new rollback content\n")
    raw = server.reindex()
    result = json.loads(raw)

    assert result["errors"] == 1
    assert result["source_errors"][0]["stage"] == "replace_source"
    assert "SECRET" not in raw
    assert source_snapshot(server._conn, "docs/note.md") == before
    assert (
        memory_db.keyword_search(
            server._conn, "new", retrieval_filter=RetrievalFilter(owner=TEST_OWNER)
        )
        == []
    )
    assert memory_db.keyword_search(
        server._conn, "old", retrieval_filter=RetrievalFilter(owner=TEST_OWNER)
    )


def test_two_forced_unchanged_runs_keep_ids_counts_and_skip_embedding(monkeypatch, tmp_path):
    import memory_v3.server as server

    vault = tmp_path / "vault"
    write(vault / "docs/stable.md", long_note("stablemarker"))
    configure(monkeypatch, tmp_path, vault)
    embedder = CountingEmbedder()
    factory_calls = {"count": 0}

    def factory():
        factory_calls["count"] += 1
        return embedder

    monkeypatch.setattr(server, "get_embedder", factory)
    assert json.loads(server.reindex())["errors"] == 0
    first = source_snapshot(server._conn, "docs/stable.md")
    embedder.inputs.clear()
    factory_calls["count"] = 0

    force_one = json.loads(server.reindex(force=True))
    force_two = json.loads(server.reindex(force=True))
    second = source_snapshot(server._conn, "docs/stable.md")

    assert force_one["status"] == "already_current"
    assert force_two["status"] == "already_current"
    assert second == first
    assert factory_calls["count"] == 0
    assert embedder.inputs == []
