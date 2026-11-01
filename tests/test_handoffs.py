import json
from datetime import datetime, timedelta, timezone
from importlib import import_module

import pytest

from memory_v3 import db
from memory_v3.config import reset_config

FROZEN_NOW = datetime(2026, 10, 2, 12, 30, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "configured-owner")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ASYNC", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_SENSORY", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_SURPRISE", "false")
    reset_config()
    import memory_v3.server as server

    server._conn = None
    yield
    server._conn = None
    reset_config()


def _add(conn, mock_embedding, content, **metadata):
    return db.add_memory(
        conn,
        content=content,
        embedding=mock_embedding(content),
        owner=metadata.pop("owner", "configured-owner"),
        namespace=metadata.pop("namespace", "handoff"),
        author=metadata.pop("author", "agent-a"),
        record_kind=metadata.pop("record_kind", "handoff"),
        expiry_at=metadata.pop("expiry_at", (FROZEN_NOW + timedelta(days=1)).isoformat()),
        **metadata,
    )


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_NOW if tz is not None else FROZEN_NOW.replace(tzinfo=None)


class StubEmbedder:
    def __init__(self, embedding):
        self.embedding = embedding
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.embedding


def test_direct_add_defaults_to_exact_30_day_utc_handoff(db_conn, mock_embedding, monkeypatch):
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    reset_config()
    import memory_v3.server as server

    monkeypatch.setattr(server, "datetime", FrozenDateTime)
    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(server, "get_embedder", lambda: StubEmbedder(mock_embedding("handoff")))

    result = json.loads(server.add_memory("handoff content", author="agent-a"))
    row = db_conn.execute("SELECT * FROM memories WHERE id = ?", (result["memory_id"],)).fetchone()

    assert row["owner"] == "configured-owner"
    assert row["namespace"] == "handoff"
    assert row["record_kind"] == "handoff"
    assert row["expiry_at"] == (FROZEN_NOW + timedelta(days=30)).isoformat()


def test_direct_add_bypasses_async_queue_and_does_not_drain_pending_entries(
    db_conn,
    mock_embedding,
    monkeypatch,
):
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ASYNC", "true")
    monkeypatch.setenv("MEMORY_V3_ASYNC_ENABLED", "true")
    reset_config()
    import memory_v3.async_ops.write_queue as write_queue
    import memory_v3.server as server

    pending_payloads = [
        {"operation": "update_memory", "memory_id": 123},
        {"operation": "archive_memory", "memory_id": 456},
    ]
    for payload in pending_payloads:
        db.enqueue_write(db_conn, payload["operation"], payload)
    before_queue = [
        dict(row)
        for row in db_conn.execute(
            "SELECT id, operation, status, payload FROM write_queue ORDER BY id"
        ).fetchall()
    ]

    def fail_queue_call(*args, **kwargs):
        raise AssertionError("direct handoff add must not touch async queue")

    monkeypatch.setattr(server, "datetime", FrozenDateTime)
    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(server, "get_embedder", lambda: StubEmbedder(mock_embedding("handoff")))
    monkeypatch.setattr(write_queue, "enqueue", fail_queue_call)
    monkeypatch.setattr(write_queue, "flush", fail_queue_call)

    result = json.loads(server.add_memory("handoff while queue has old mutations"))

    after_queue = [
        dict(row)
        for row in db_conn.execute(
            "SELECT id, operation, status, payload FROM write_queue ORDER BY id"
        ).fetchall()
    ]
    row = db_conn.execute(
        "SELECT owner, namespace, record_kind, expiry_at FROM memories WHERE id = ?",
        (result["memory_id"],),
    ).fetchone()

    assert after_queue == before_queue
    assert row["owner"] == "configured-owner"
    assert row["namespace"] == "handoff"
    assert row["record_kind"] == "handoff"
    assert row["expiry_at"] == (FROZEN_NOW + timedelta(days=30)).isoformat()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"namespace": "business"},
    ],
)
def test_direct_add_rejects_custom_namespace_before_side_effects(monkeypatch, kwargs):
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    reset_config()
    import memory_v3.server as server

    calls = []

    def fail(*args, **call_kwargs):
        calls.append((args, call_kwargs))
        raise AssertionError("direct semantics gate must run before providers and DB")

    monkeypatch.setattr(server, "_get_conn", fail)
    monkeypatch.setattr(server, "get_embedder", fail)
    monkeypatch.setattr(server.db, "add_memory", fail)

    result = json.loads(server.add_memory("blocked", **kwargs))

    assert result["code"] == "unsupported_direct_write_semantics"
    assert result["memory_id"] is None
    assert calls == []


def test_direct_add_accepts_durable_and_admin_override(monkeypatch):
    """durable=True and admin_override=True now pass the direct-write gate."""
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    reset_config()
    import memory_v3.server as server

    for kwargs in [{"durable": True}, {"admin_override": True}, {"durable": True, "admin_override": True}]:
        calls = []

        def fake_get_conn():
            calls.append("get_conn")
            return object()

        class FakeEmbedder:
            def embed(self, text):
                calls.append("embed")
                return [0.0] * 1536

        monkeypatch.setattr(server, "_get_conn", fake_get_conn)
        monkeypatch.setattr(server, "get_embedder", lambda: FakeEmbedder())
        monkeypatch.setattr(server.db, "add_memory", lambda *a, **k: calls.append("add_memory") or 999)

        result = json.loads(server.add_memory("test fact", **kwargs))

        # durable/admin_override pass the metadata gate (not rejected)
        assert result.get("code") != "unsupported_direct_write_semantics"
        assert "get_conn" in calls  # gate was passed


def test_direct_extract_accepts_durable_past_gate(monkeypatch):
    """durable=True now passes the direct-write gate in extract_from_conversation."""
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    reset_config()
    import memory_v3.server as server

    calls = []

    def fake_get_conn():
        calls.append("get_conn")
        return object()

    class FakeEmbedder:
        def embed(self, text):
            calls.append("embed")
            return [0.0] * 1536

    monkeypatch.setattr(server, "_get_conn", fake_get_conn)
    monkeypatch.setattr(server, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(server.db, "add_memory", lambda *a, **k: calls.append("add_memory") or 1)

    result = json.loads(server.extract_from_conversation("test", durable=True))

    # durable passes the metadata gate (not rejected)
    assert result.get("code") != "unsupported_direct_write_semantics"
    assert "get_conn" in calls  # gate was passed


@pytest.mark.parametrize(
    ("tool_name", "call"),
    [
        ("update", lambda server: server.update(1, content="new")),
        ("forget", lambda server: server.forget(1, reason="test")),
        ("compact_text", lambda server: server.compact_text("text")),
    ],
)
def test_legacy_direct_mutation_tools_are_stable_stubs_before_side_effects(
    monkeypatch,
    tool_name,
    call,
):
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    reset_config()
    import memory_v3.server as server

    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(f"{tool_name} must not open config, DB, or provider")

    monkeypatch.setattr(server, "get_config", fail)
    monkeypatch.setattr(server, "_get_conn", fail)
    monkeypatch.setattr(server, "get_embedder", fail)
    monkeypatch.setattr(server.db, "update_memory", fail)
    monkeypatch.setattr(server.db, "archive_memory", fail)

    result = json.loads(call(server))

    assert result["code"] == "direct_mutation_disabled"
    assert result["deprecated"] is True
    assert result["mutated"] is False
    assert tool_name in result["error"]
    assert calls == []


def test_direct_extract_is_handoff_add_only_without_graph_behavior(
    db_conn,
    mock_embedding,
    monkeypatch,
):
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ZETTELKASTEN", "true")
    monkeypatch.setenv("MEMORY_V3_GRAPH_MUTATION_ENABLED", "true")
    reset_config()
    existing = _add(db_conn, mock_embedding, "existing handoff")
    import memory_v3.graphs.zettelkasten as zettelkasten
    extraction = import_module("memory_v3.lifecycle.extraction")
    import memory_v3.server as server

    class StubLLM:
        def chat(self, prompt, max_tokens=2048):
            return "- replacement handoff content"

    calls = []
    monkeypatch.setattr(server, "datetime", FrozenDateTime)
    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(extraction, "get_llm", lambda: StubLLM())
    monkeypatch.setattr(extraction, "embed_with_cache", lambda text: mock_embedding(text))
    monkeypatch.setattr(
        extraction,
        "decide_action",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must stay add-only")),
    )
    monkeypatch.setattr(
        zettelkasten,
        "auto_link",
        lambda *args, **kwargs: calls.append("auto_link"),
    )
    monkeypatch.setattr(
        server,
        "_get_graph_manager",
        lambda: calls.append("graph_manager"),
    )

    result = json.loads(server.extract_from_conversation("replace it"))

    assert result["added"] == 1
    assert result["updated"] == 0
    assert result["deleted"] == 0
    archived = db_conn.execute(
        "SELECT archived FROM memories WHERE id = ?",
        (existing,),
    ).fetchone()[0]
    inserted = db_conn.execute("SELECT * FROM memories ORDER BY id DESC LIMIT 1").fetchone()
    assert archived == 0
    assert inserted["record_kind"] == "handoff"
    assert inserted["namespace"] == "handoff"
    assert inserted["expiry_at"] == (FROZEN_NOW + timedelta(days=30)).isoformat()
    assert calls == []


def test_list_active_handoffs_filters_owner_agent_since_expiry_and_provenance(
    db_conn,
    mock_embedding,
):
    keep = _add(
        db_conn,
        mock_embedding,
        "active handoff",
        author="agent-a",
        source_file="projects/demo.md",
        source_hash="abc123",
    )
    db_conn.execute(
        "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
        ((FROZEN_NOW - timedelta(hours=1)).isoformat(), FROZEN_NOW.isoformat(), keep),
    )
    _add(db_conn, mock_embedding, "expired", expiry_at=FROZEN_NOW.isoformat())
    _add(db_conn, mock_embedding, "future other agent", author="agent-b")
    _add(db_conn, mock_embedding, "cross owner", owner="other-owner")
    _add(
        db_conn,
        mock_embedding,
        "derived",
        namespace="project:demo",
        record_kind="derived",
        expiry_at=None,
    )
    archived = _add(db_conn, mock_embedding, "archived")
    db_conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (archived,))
    db_conn.commit()

    rows = db.list_active_handoffs(
        db_conn,
        agent_id="agent-a",
        since=FROZEN_NOW - timedelta(hours=2),
        now=FROZEN_NOW,
    )

    assert [row["id"] for row in rows] == [keep]
    row = rows[0]
    for key in (
        "source_file",
        "source_hash",
        "owner",
        "namespace",
        "created_at",
        "updated_at",
        "verification_status",
        "record_kind",
        "expiry_at",
        "author",
    ):
        assert key in row


def test_sweeper_disabled_returns_before_db_open(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_HANDOFF_SWEEPER_ENABLED", "false")
    reset_config()
    import memory_v3.server as server

    monkeypatch.setattr(
        server,
        "_get_conn",
        lambda: (_ for _ in ()).throw(AssertionError("disabled sweeper opened DB")),
    )

    result = json.loads(server.sweep_expired_handoffs(now=FROZEN_NOW.isoformat()))

    assert result == {"deleted": 0, "enabled": False}


def test_sweep_removes_only_expired_owner_handoffs_and_scrubs_links(
    db_conn,
    mock_embedding,
    monkeypatch,
):
    monkeypatch.setenv("MEMORY_V3_HANDOFF_SWEEPER_ENABLED", "true")
    reset_config()
    expired = _add(
        db_conn,
        mock_embedding,
        "expired owner handoff",
        expiry_at=FROZEN_NOW.isoformat(),
    )
    future = _add(db_conn, mock_embedding, "future owner handoff")
    derived = _add(
        db_conn,
        mock_embedding,
        "derived old row",
        namespace="project:demo",
        record_kind="derived",
        expiry_at=(FROZEN_NOW - timedelta(days=1)).isoformat(),
    )
    cross_owner = _add(
        db_conn,
        mock_embedding,
        "cross owner expired",
        owner="other-owner",
        expiry_at=(FROZEN_NOW - timedelta(days=1)).isoformat(),
    )
    malformed = _add(
        db_conn,
        mock_embedding,
        "malformed expiry",
        expiry_at="not-a-timestamp",
    )
    naive = _add(
        db_conn,
        mock_embedding,
        "naive expiry",
        expiry_at="2026-10-02T12:29:59",
    )
    db_conn.execute(
        "UPDATE memories SET linked_memories = ? WHERE id = ?",
        (json.dumps([{"memory_id": expired}, {"memory_id": derived}]), future),
    )
    cross_owner_links = json.dumps([{"memory_id": expired}, {"memory_id": derived}])
    cross_owner_updated_at = "2026-10-02T00:00:00+00:00"
    db_conn.execute(
        "UPDATE memories SET linked_memories = ?, updated_at = ? WHERE id = ?",
        (cross_owner_links, cross_owner_updated_at, cross_owner),
    )
    db_conn.commit()
    import memory_v3.server as server

    monkeypatch.setattr(server, "_conn", db_conn)
    result = json.loads(server.sweep_expired_handoffs(now=FROZEN_NOW.isoformat()))

    assert result["deleted"] == 1
    ids = {row[0] for row in db_conn.execute("SELECT id FROM memories")}
    assert expired not in ids
    assert {future, derived, cross_owner, malformed, naive} <= ids
    assert db_conn.execute("SELECT 1 FROM memory_vec WHERE id = ?", (expired,)).fetchone() is None
    assert (
        db_conn.execute("SELECT 1 FROM memory_fts WHERE rowid = ?", (expired,)).fetchone() is None
    )
    links = json.loads(
        db_conn.execute(
            "SELECT linked_memories FROM memories WHERE id = ?",
            (future,),
        ).fetchone()[0]
    )
    assert links == [{"memory_id": derived}]
    cross_owner_row = db_conn.execute(
        "SELECT linked_memories, updated_at FROM memories WHERE id = ?",
        (cross_owner,),
    ).fetchone()
    assert cross_owner_row["linked_memories"] == cross_owner_links
    assert cross_owner_row["updated_at"] == cross_owner_updated_at


def test_sweeper_failure_rolls_back_memory_vector_fts_and_links(
    db_conn,
    mock_embedding,
    monkeypatch,
):
    monkeypatch.setenv("MEMORY_V3_HANDOFF_SWEEPER_ENABLED", "true")
    reset_config()
    expired = _add(
        db_conn,
        mock_embedding,
        "rollbacktoken expired handoff",
        expiry_at=FROZEN_NOW.isoformat(),
    )
    survivor = _add(db_conn, mock_embedding, "surviving handoff")
    derived = _add(
        db_conn,
        mock_embedding,
        "derived reference",
        namespace="project:demo",
        record_kind="derived",
        expiry_at=None,
    )
    original_links = [{"memory_id": expired}, {"memory_id": derived}]
    db_conn.execute(
        "UPDATE memories SET linked_memories = ? WHERE id = ?",
        (json.dumps(original_links), survivor),
    )
    db_conn.commit()

    def fail_after_delete(conn, memory_ids):
        assert memory_ids == [expired]
        raise RuntimeError("SECRET_SENTINEL /tmp/leak SELECT * FROM memories")

    monkeypatch.setattr(
        db.sweep_expired_handoffs,
        "test_hook",
        fail_after_delete,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="SECRET_SENTINEL"):
        db.sweep_expired_handoffs(db_conn, now=FROZEN_NOW)

    assert db_conn.execute("SELECT 1 FROM memories WHERE id = ?", (expired,)).fetchone() is not None
    assert (
        db_conn.execute("SELECT 1 FROM memory_vec WHERE id = ?", (expired,)).fetchone() is not None
    )
    assert (
        db_conn.execute(
            "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
            ("rollbacktoken",),
        ).fetchone()[0]
        == expired
    )
    links = json.loads(
        db_conn.execute(
            "SELECT linked_memories FROM memories WHERE id = ?",
            (survivor,),
        ).fetchone()[0]
    )
    assert links == original_links


def test_list_active_handoffs_public_failure_is_sanitized(monkeypatch):
    import memory_v3.server as server

    secret = "SECRET_SENTINEL /home/base/private.db SELECT password FROM tokens"
    monkeypatch.setattr(
        server.db,
        "list_active_handoffs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr(server, "_get_conn", lambda: object())

    raw = server.list_active_handoffs(agent_id="agent-a")
    result = json.loads(raw)

    assert result == {
        "error": "List active handoffs failed",
        "code": "handoff_operation_failed",
        "count": 0,
        "handoffs": [],
    }
    assert "SECRET_SENTINEL" not in raw
    assert "/home/base" not in raw
    assert "SELECT" not in raw
    assert "password" not in raw


def test_sweep_expired_handoffs_public_failure_is_sanitized(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_HANDOFF_SWEEPER_ENABLED", "true")
    reset_config()
    import memory_v3.server as server

    secret = "SECRET_SENTINEL /home/base/private.db DELETE FROM memories"
    monkeypatch.setattr(
        server.db,
        "sweep_expired_handoffs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr(server, "_get_conn", lambda: object())

    raw = server.sweep_expired_handoffs(now=FROZEN_NOW.isoformat())
    result = json.loads(raw)

    assert result == {
        "error": "Sweep expired handoffs failed",
        "code": "handoff_operation_failed",
        "enabled": True,
        "deleted": 0,
    }
    assert "SECRET_SENTINEL" not in raw
    assert "/home/base" not in raw
    assert "DELETE" not in raw


def test_agent_sync_is_deprecated_active_handoffs_alias_without_changelog(
    db_conn,
    mock_embedding,
    monkeypatch,
):
    handoff = _add(db_conn, mock_embedding, "sync handoff", author="agent-a")
    import memory_v3.server as server

    monkeypatch.setattr(db, "datetime", FrozenDateTime)
    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("changelog opened")),
    )

    result = json.loads(server.agent_sync("agent-a"))

    assert result["deprecated"] is True
    assert result["mode"] == "active_handoffs"
    assert result["compatibility_mode"] is False
    assert "entries" not in result
    assert [row["id"] for row in result["handoffs"]] == [handoff]


def test_agent_sync_ignores_removed_legacy_flag(db_conn, mock_embedding, monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LEGACY_AGENT_SYNC_ENABLED", "true")
    reset_config()
    handoff = _add(db_conn, mock_embedding, "sync handoff", author="agent-a")
    import memory_v3.server as server

    monkeypatch.setattr(db, "datetime", FrozenDateTime)
    monkeypatch.setattr(server, "_conn", db_conn)

    result = json.loads(server.agent_sync("agent-a"))

    assert result["mode"] == "active_handoffs"
    assert result["compatibility_mode"] is False
    assert [row["id"] for row in result["handoffs"]] == [handoff]


def test_multi_agent_legacy_write_and_offset_surfaces_are_absent():
    import memory_v3.multi_agent as multi_agent

    for name in (
        "get_offset",
        "set_offset",
        "authority_write",
        "legacy_changelog_sync",
        "get_unread_changelog",
        "get_authority_level",
        "can_overwrite",
        "log_conflict",
        "resolve_conflict",
        "detect_conflict",
    ):
        assert not hasattr(multi_agent, name)


def test_agent_sync_failure_is_sanitized(db_conn, monkeypatch):
    import memory_v3.multi_agent as multi_agent
    import memory_v3.server as server

    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(
        multi_agent,
        "sync_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("SECRET_SENTINEL /tmp/path")),
    )

    raw = server.agent_sync("agent-a")
    result = json.loads(raw)

    assert result["error"] == "Agent synchronization failed"
    assert result["mode"] == "active_handoffs"
    assert result["compatibility_mode"] is False
    assert "SECRET_SENTINEL" not in raw
    assert "/tmp/path" not in raw


def test_direct_add_never_auto_links_when_graph_config_enabled(
    db_conn,
    mock_embedding,
    monkeypatch,
):
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ZETTELKASTEN", "true")
    monkeypatch.setenv("MEMORY_V3_GRAPH_MUTATION_ENABLED", "true")
    reset_config()
    import memory_v3.graphs.zettelkasten as zettelkasten
    import memory_v3.server as server

    calls = []
    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(
        server,
        "get_embedder",
        lambda: StubEmbedder(mock_embedding("handoff")),
    )
    monkeypatch.setattr(
        zettelkasten,
        "auto_link",
        lambda *args, **kwargs: calls.append("auto_link"),
    )
    monkeypatch.setattr(
        server,
        "_get_graph_manager",
        lambda: calls.append("graph_manager"),
    )

    result = json.loads(server.add_memory("direct add with graph enabled"))

    assert result["memory_id"]
    assert calls == []


def test_direct_write_and_sweeper_kill_switches_are_independent(
    db_conn,
    mock_embedding,
    monkeypatch,
):
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")
    monkeypatch.setenv("MEMORY_V3_HANDOFF_SWEEPER_ENABLED", "true")
    reset_config()
    _add(db_conn, mock_embedding, "expired", expiry_at=FROZEN_NOW.isoformat())
    import memory_v3.server as server

    monkeypatch.setattr(server, "_conn", db_conn)
    assert "Direct writes are disabled" in server.add_memory("blocked")
    assert json.loads(server.sweep_expired_handoffs(now=FROZEN_NOW.isoformat()))["deleted"] == 1

    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_HANDOFF_SWEEPER_ENABLED", "false")
    reset_config()
    monkeypatch.setattr(
        server,
        "get_embedder",
        lambda: StubEmbedder(mock_embedding("allowed")),
    )
    assert json.loads(server.add_memory("allowed"))["memory_id"]
    assert json.loads(server.sweep_expired_handoffs(now=FROZEN_NOW.isoformat()))["deleted"] == 0


def test_direct_write_and_sweeper_disabled_independently_make_zero_changes(
    db_conn,
    mock_embedding,
    monkeypatch,
):
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")
    monkeypatch.setenv("MEMORY_V3_HANDOFF_SWEEPER_ENABLED", "false")
    reset_config()
    expired = _add(db_conn, mock_embedding, "expired", expiry_at=FROZEN_NOW.isoformat())
    before = db_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    import memory_v3.server as server

    monkeypatch.setattr(server, "_conn", db_conn)
    add_result = json.loads(server.add_memory("blocked"))
    sweep_result = json.loads(server.sweep_expired_handoffs(now=FROZEN_NOW.isoformat()))
    after = db_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    assert add_result["error"] == "Direct writes are disabled by MEMORY_V3_DIRECT_WRITES_ENABLED"
    assert sweep_result == {"enabled": False, "deleted": 0}
    assert after == before
    assert db_conn.execute("SELECT 1 FROM memories WHERE id = ?", (expired,)).fetchone() is not None
