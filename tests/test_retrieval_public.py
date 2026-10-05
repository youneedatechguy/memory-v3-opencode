import json
from types import SimpleNamespace

import pytest

from memory_v3 import db, server
from memory_v3.retrieval import search as retrieval_search
from memory_v3.retrieval import stage1_coarse, stage2_fine, stage3_organize
from memory_v3.retrieval.filters import RetrievalFilter

PROVENANCE = {
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
}
REMOVED_PROVENANCE = {
    "source_generation_id",
    "source_generation_status",
    "line_start",
    "line_end",
    "heading_path",
    "rank_marker",
    "rank_weight",
    "historical",
    "deleted_at",
}


class Embedder:
    def __init__(self, embed):
        self._embed = embed
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return self._embed(text)


def _cfg(*, three_stage=False, owner="alice"):
    return SimpleNamespace(
        source_policy_owner=owner,
        enable_three_stage=three_stage,
        enable_structure_tags=False,
        stage1_max_candidates=50,
        stage1_community_top=3,
        rrf_k=60,
    )


def _add(conn, embed, content, **metadata):
    return db.add_memory(
        conn,
        content=content,
        embedding=embed(content),
        source_file=metadata.pop("source_file", "projects/alpha/doc.md"),
        source_line=metadata.pop("source_line", 7),
        source_hash=metadata.pop("source_hash", f"hash-{content}"),
        owner=metadata.pop("owner", "alice"),
        namespace=metadata.pop("namespace", "project:alpha"),
        author=metadata.pop("author", "morgan"),
        sensitivity=metadata.pop("sensitivity", "private"),
        record_kind=metadata.pop("record_kind", "derived"),
        source_id=metadata.pop("source_id", f"source-{content}"),
        chunk_id=metadata.pop("chunk_id", f"chunk-{content}"),
        chunk_index=metadata.pop("chunk_index", 0),
        chunk_count=metadata.pop("chunk_count", 1),
        source_updated_at=metadata.pop("source_updated_at", "2026-09-01T00:00:00+00:00"),
        project_status=metadata.pop("project_status", "active"),
        project_updated_at=metadata.pop("project_updated_at", "2026-09-02T00:00:00+00:00"),
        **metadata,
    )


def _archive(conn, memory_id):
    conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,))
    conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
    conn.commit()


def _use_server(monkeypatch, conn, embed, cfg):
    provider = Embedder(embed)
    monkeypatch.setattr(server, "get_config", lambda: cfg)
    monkeypatch.setattr(server, "_get_conn", lambda: conn)
    monkeypatch.setattr(server, "get_embedder", lambda: provider)
    monkeypatch.setattr(server, "_get_embedder", lambda: provider)
    return provider


def _payload(raw):
    return json.loads(raw)


def test_public_config_owner_gate_runs_before_db_provider_or_graph(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "get_config", lambda: _cfg(owner=""))
    monkeypatch.setattr(server, "_get_conn", lambda: calls.append("db"))
    monkeypatch.setattr(server, "get_embedder", lambda: calls.append("embed"))
    monkeypatch.setattr(server, "_get_graph_manager", lambda: calls.append("graph"))

    endpoints = (
        lambda: server.search("needle"),
        lambda: server.keyword_search("needle"),
        lambda: server.get(1),
        server.list_topics,
        server.list_recent,
    )

    for endpoint in endpoints:
        assert _payload(endpoint())["code"] == "invalid_retrieval_filter"
    assert calls == []


@pytest.mark.parametrize("fn", [db.keyword_search, db.hybrid_search])
def test_db_public_search_helpers_reject_missing_or_ownerless_filters(db_conn, mock_embedding, fn):
    _add(db_conn, mock_embedding, "ownerless leak needle")

    with pytest.raises(ValueError, match="owner"):
        if fn is db.hybrid_search:
            fn(db_conn, "needle", mock_embedding("needle"), retrieval_filter=None)
        else:
            fn(db_conn, "needle", retrieval_filter=None)

    with pytest.raises(ValueError, match="owner"):
        if fn is db.hybrid_search:
            fn(db_conn, "needle", mock_embedding("needle"), retrieval_filter=RetrievalFilter())
        else:
            fn(db_conn, "needle", retrieval_filter=RetrievalFilter())


def test_ownerless_hybrid_search_raises_before_sql_or_search_work(mock_embedding):
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
            conn, "needle", mock_embedding("needle"), retrieval_filter=RetrievalFilter()
        )

    assert conn.calls == 0


def test_hybrid_search_ranks_all_eligible_vectors_without_global_neighbor_starvation(db_conn):
    now = "2026-10-02T00:00:00+00:00"
    query_embedding = [0.0] * 768
    close_other_embedding = db._serialize_f32([0.0] * 768)
    far_eligible_embedding = db._serialize_f32([1.0] * 768)

    def insert_row(index, owner, content, embedding):
        cursor = db_conn.execute(
            """INSERT INTO memories
            (content, content_type, source_file, author, confidence, protected, tags,
             created_at, updated_at, last_accessed_at, importance_score, decay_rate,
             content_hash, governance_layer, structure_type, surprise_score, source_hash,
             owner, namespace, sensitivity, record_kind, source_id, chunk_id,
             chunk_index, chunk_count, source_updated_at, project_status, project_updated_at)
            VALUES (?, 'fact', ?, 'morgan', 0.95, 0, '[]', ?, ?, ?, 0.5, 0.1,
                    ?, 3, 'narrative', 0.5, ?, ?, 'docs', 'private', 'derived', ?, ?,
                    0, 1, ?, 'active', ?)""",
            (
                content,
                f"docs/{owner}-{index}.md",
                now,
                now,
                now,
                f"hash-{owner}-{index}",
                f"source-hash-{owner}-{index}",
                owner,
                f"source-{owner}-{index}",
                f"chunk-{owner}-{index}",
                now,
                now,
            ),
        )
        db_conn.execute(
            "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
            (cursor.lastrowid, embedding),
        )
        return cursor.lastrowid

    for index in range(4100):
        insert_row(index, "bob", f"bob close vector {index}", close_other_embedding)
    wanted = insert_row(0, "alice", "alice far vector", far_eligible_embedding)
    db_conn.commit()

    rows = db.hybrid_search(
        db_conn,
        "nomatchstarvation",
        query_embedding,
        limit=1,
        retrieval_filter=RetrievalFilter(owner="alice", namespaces=("docs",)),
    )

    assert [row["id"] for row in rows] == [wanted]
    assert rows[0]["owner"] == "alice"


@pytest.mark.parametrize("three_stage", [True, False])
def test_one_owner_filter_reaches_every_search_stage_and_fallback(
    db_conn, mock_embedding, monkeypatch, three_stage
):
    filt = RetrievalFilter(
        owner="alice",
        namespaces=("project:alpha", "docs"),
        project_slug="alpha",
        path_prefix="projects/alpha/",
        author="morgan",
        sensitivity="private",
        record_kind="derived",
    )
    seen = []

    def candidates(conn, query, embedding, cfg, retrieval_filter=None):
        seen.append(("stage1", retrieval_filter))
        return [1]

    def score(conn, ids, query, embedding, cfg, retrieval_filter=None):
        seen.append(("stage2", retrieval_filter))
        return [{"id": 1, "content": "needle"}]

    def organize(conn, rows, query, cfg, retrieval_filter=None):
        seen.append(("stage3", retrieval_filter))
        return rows

    def fallback(conn, query, embedding, **kwargs):
        seen.append(("fallback", kwargs["retrieval_filter"]))
        return []

    monkeypatch.setattr(stage1_coarse, "get_candidates", candidates)
    monkeypatch.setattr(stage2_fine, "score_candidates", score)
    monkeypatch.setattr(stage3_organize, "organize", organize)
    monkeypatch.setattr(db, "hybrid_search", fallback)

    retrieval_search(
        db_conn,
        "needle",
        mock_embedding("needle"),
        config=_cfg(three_stage=three_stage),
        retrieval_filter=filt,
    )

    assert [name for name, _ in seen] == (
        ["stage1", "stage2", "stage3"] if three_stage else ["fallback"]
    )
    assert all(value is filt for _, value in seen)


def test_public_search_keyword_get_recent_topics_owner_scope_and_v4_provenance(
    db_conn, mock_embedding, monkeypatch
):
    wanted = _add(db_conn, mock_embedding, "public filter needle")
    _add(db_conn, mock_embedding, "public filter wrong owner", owner="bob")
    _add(db_conn, mock_embedding, "public filter wrong namespace", namespace="project:beta")
    _add(db_conn, mock_embedding, "public filter wrong path", source_file="projects/beta/doc.md")
    _add(db_conn, mock_embedding, "public filter wrong author", author="casey")
    _add(db_conn, mock_embedding, "public filter wrong sensitivity", sensitivity="normal")
    _add(db_conn, mock_embedding, "public filter wrong kind", record_kind="handoff")
    _use_server(monkeypatch, db_conn, mock_embedding, _cfg())
    kwargs = {
        "namespace": "project:alpha",
        "namespaces": ["project:alpha"],
        "project_slug": "alpha",
        "path_prefix": "projects/alpha/",
        "author": "morgan",
        "sensitivity": "private",
        "record_kind": "derived",
    }

    search_rows = _payload(server.search("public filter", limit=20, **kwargs))["results"]
    keyword_rows = _payload(server.keyword_search("public", limit=20, **kwargs))["results"]
    recent_rows = _payload(server.list_recent(hours=24 * 366, limit=20))["memories"]
    got = _payload(server.get(wanted, **kwargs))
    topics = _payload(server.list_topics())["topics"]

    assert [row["id"] for row in search_rows] == [wanted]
    assert [row["id"] for row in keyword_rows] == [wanted]
    assert wanted in [row["id"] for row in recent_rows]
    assert got["id"] == wanted
    assert topics == [{"content_type": "fact", "count": 6, "source": "content_type"}]
    for row in (search_rows[0], keyword_rows[0], got):
        assert PROVENANCE <= row.keys()
        assert REMOVED_PROVENANCE.isdisjoint(row.keys())
        assert row["owner"] == "alice"
        assert row["namespace"] == "project:alpha"
        assert row["source_id"]
        assert row["chunk_id"]
        assert row["source_line"] == 7
        assert "related_nodes" not in row
        assert "see_also" not in row


def test_cross_owner_terminal_and_archived_rows_are_absent_without_content_leak(
    db_conn, mock_embedding, monkeypatch
):
    cross_owner = _add(db_conn, mock_embedding, "cross owner secret", owner="bob")
    _add(db_conn, mock_embedding, "terminal secret", project_status="completed")
    archived = _add(db_conn, mock_embedding, "archived secret")
    _archive(db_conn, archived)
    _use_server(monkeypatch, db_conn, mock_embedding, _cfg(owner="alice"))

    payload = json.dumps(
        {
            "search": _payload(server.search("secret", limit=20)),
            "keyword": _payload(server.keyword_search("secret", limit=20)),
            "get": _payload(server.get(cross_owner)),
        }
    )

    assert _payload(server.search("secret", limit=20))["results"] == []
    assert _payload(server.keyword_search("secret", limit=20))["results"] == []
    assert "not found" in _payload(server.get(cross_owner))["error"].lower()
    assert "cross owner secret" not in payload
    assert "terminal secret" not in payload
    assert "archived secret" not in payload


def test_historical_argument_is_removed_from_public_api(db_conn, mock_embedding, monkeypatch):
    _add(db_conn, mock_embedding, "ordinary needle")
    _use_server(monkeypatch, db_conn, mock_embedding, _cfg())

    with pytest.raises(TypeError):
        server.keyword_search("ordinary", historical=True)
    with pytest.raises(TypeError):
        server.search("ordinary", historical=True)
    with pytest.raises(TypeError):
        server.get(1, historical=True)


def test_stage3_never_adds_neighbors_and_preserves_input_ranking(db_conn, mock_embedding):
    first = _add(db_conn, mock_embedding, "first ranked")
    second = _add(db_conn, mock_embedding, "second ranked")
    rows = [
        dict(db_conn.execute("SELECT * FROM memories WHERE id = ?", (second,)).fetchone()),
        dict(db_conn.execute("SELECT * FROM memories WHERE id = ?", (first,)).fetchone()),
    ]

    results = stage3_organize.organize(
        db_conn,
        rows,
        "ranked",
        _cfg(three_stage=True),
        RetrievalFilter(owner="alice"),
    )

    assert [row["id"] for row in results] == [second, first]
    assert all(row["related_nodes"] == [] for row in results)
    assert all(row["see_also"] == [] for row in results)


def test_removed_columns_are_not_required_by_public_sql(db_conn, mock_embedding, monkeypatch):
    _add(db_conn, mock_embedding, "removed column needle")
    _use_server(monkeypatch, db_conn, mock_embedding, _cfg())

    memory_columns = {row[1] for row in db_conn.execute("PRAGMA table_info(memories)")}
    assert REMOVED_PROVENANCE.isdisjoint(memory_columns)
    assert _payload(server.search("needle"))["results"]
    assert _payload(server.keyword_search("needle"))["results"]
