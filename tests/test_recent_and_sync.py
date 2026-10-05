import json
from datetime import datetime, timedelta, timezone

import pytest

from memory_v3 import db
from memory_v3.config import reset_config
from memory_v3.retrieval.filters import RetrievalFilter

FROZEN_NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


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


def _filter(**kwargs):
    return RetrievalFilter(owner=kwargs.pop("owner", "configured-owner"), **kwargs)


def _add(conn, mock_embedding, content, *, created_at=None, **metadata):
    memory_id = db.add_memory(
        conn,
        content=content,
        embedding=mock_embedding(content),
        owner=metadata.pop("owner", "configured-owner"),
        namespace=metadata.pop("namespace", "project:demo"),
        author=metadata.pop("author", "agent-a"),
        record_kind=metadata.pop("record_kind", "derived"),
        expiry_at=metadata.pop("expiry_at", None),
        **metadata,
    )
    if created_at is not None:
        conn.execute(
            """
            UPDATE memories
            SET created_at = ?, updated_at = ?, last_accessed_at = ?
            WHERE id = ?
            """,
            (created_at, created_at, created_at, memory_id),
        )
        conn.commit()
    return memory_id


def _contents(rows):
    return [row["content"] for row in rows]


def test_list_recent_requires_configured_owner_filter(db_conn):
    with pytest.raises(ValueError, match="retrieval owner is required"):
        db.list_recent(db_conn, hours=1, limit=10, now=FROZEN_NOW)
    with pytest.raises(ValueError, match="retrieval owner is required"):
        db.list_recent(
            db_conn,
            hours=1,
            limit=10,
            now=FROZEN_NOW,
            retrieval_filter=RetrievalFilter(),
        )


def test_list_recent_applies_true_utc_cutoff_and_includes_exact_boundary(
    db_conn,
    mock_embedding,
):
    _add(
        db_conn,
        mock_embedding,
        "two hours old",
        created_at=(FROZEN_NOW - timedelta(hours=2)).isoformat(),
    )
    boundary = _add(
        db_conn,
        mock_embedding,
        "exactly one hour old",
        created_at=(FROZEN_NOW - timedelta(hours=1)).isoformat(),
    )
    fresh = _add(db_conn, mock_embedding, "fresh", created_at=FROZEN_NOW.isoformat())

    rows = db.list_recent(
        db_conn,
        hours=1,
        limit=10,
        now=FROZEN_NOW,
        retrieval_filter=_filter(),
    )

    assert [row["id"] for row in rows] == [fresh, boundary]
    assert "two hours old" not in _contents(rows)


def test_list_recent_normalizes_timezone_offsets_and_excludes_malformed(
    db_conn,
    mock_embedding,
):
    _add(
        db_conn,
        mock_embedding,
        "offset inside",
        created_at="2026-10-02T22:30:00+11:00",
    )
    _add(
        db_conn,
        mock_embedding,
        "offset outside",
        created_at="2026-10-02T21:59:59+11:00",
    )
    malformed = _add(db_conn, mock_embedding, "malformed")
    naive = _add(db_conn, mock_embedding, "naive")
    db_conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("not-a-timestamp", malformed),
    )
    db_conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("2026-10-02T11:30:00", naive),
    )
    db_conn.commit()

    rows = db.list_recent(
        db_conn,
        hours=1,
        limit=10,
        now=FROZEN_NOW,
        retrieval_filter=_filter(),
    )

    assert _contents(rows) == ["offset inside"]


def test_list_recent_filters_owner_status_archive_and_expired_handoffs(
    db_conn,
    mock_embedding,
):
    derived = _add(
        db_conn,
        mock_embedding,
        "eligible derived",
        created_at=FROZEN_NOW.isoformat(),
    )
    handoff = _add(
        db_conn,
        mock_embedding,
        "active handoff",
        created_at=FROZEN_NOW.isoformat(),
        namespace="handoff",
        record_kind="handoff",
        expiry_at=(FROZEN_NOW + timedelta(hours=1)).isoformat(),
    )
    excluded = [
        _add(
            db_conn,
            mock_embedding,
            "cross owner",
            created_at=FROZEN_NOW.isoformat(),
            owner="other",
        ),
        _add(
            db_conn,
            mock_embedding,
            "terminal",
            created_at=FROZEN_NOW.isoformat(),
            project_status="completed",
        ),
        _add(
            db_conn,
            mock_embedding,
            "expired handoff",
            created_at=FROZEN_NOW.isoformat(),
            namespace="handoff",
            record_kind="handoff",
            expiry_at=FROZEN_NOW.isoformat(),
        ),
        _add(
            db_conn,
            mock_embedding,
            "malformed expiry handoff",
            created_at=FROZEN_NOW.isoformat(),
            namespace="handoff",
            record_kind="handoff",
            expiry_at="not-a-timestamp",
        ),
        _add(
            db_conn,
            mock_embedding,
            "naive expiry handoff",
            created_at=FROZEN_NOW.isoformat(),
            namespace="handoff",
            record_kind="handoff",
            expiry_at="2026-10-02T13:00:00",
        ),
    ]
    archived = _add(db_conn, mock_embedding, "archived", created_at=FROZEN_NOW.isoformat())
    db_conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (archived,))
    db_conn.commit()

    rows = db.list_recent(db_conn, hours=1, limit=20, now=FROZEN_NOW, retrieval_filter=_filter())

    assert [row["id"] for row in rows] == [handoff, derived]
    assert not ({row["id"] for row in rows} & set(excluded + [archived]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hours": 0},
        {"hours": -1},
        {"hours": True},
        {"limit": 0},
        {"limit": -1},
        {"limit": 1001},
        {"limit": False},
    ],
)
def test_list_recent_rejects_invalid_bounds(db_conn, kwargs):
    with pytest.raises(ValueError):
        db.list_recent(db_conn, now=FROZEN_NOW, retrieval_filter=_filter(), **kwargs)


def test_public_list_recent_returns_truthful_v4_provenance(db_conn, mock_embedding, monkeypatch):
    memory_id = _add(
        db_conn,
        mock_embedding,
        "with provenance",
        created_at=FROZEN_NOW.isoformat(),
        source_file="projects/demo.md",
        source_hash="abc123",
        source_id="source-1",
        chunk_id="chunk-1",
        chunk_index=0,
        chunk_count=2,
        source_line=7,
        source_updated_at="2026-10-02T10:00:00+00:00",
    )
    import memory_v3.server as server

    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(
        server.db,
        "datetime",
        type("FrozenDateTime", (datetime,), {"now": classmethod(lambda cls, tz=None: FROZEN_NOW)}),
    )

    result = json.loads(server.list_recent(hours=1, limit=5))

    row = result["memories"][0]
    assert result["count"] == 1
    assert row["id"] == memory_id
    assert row["source_file"] == "projects/demo.md"
    assert row["path"] == "projects/demo.md"
    assert row["source_hash"] == "abc123"
    assert row["source_id"] == "source-1"
    assert row["chunk_id"] == "chunk-1"
    assert row["chunk_index"] == 0
    assert row["chunk_count"] == 2
    assert row["source_line"] == 7
    assert row["owner"] == "configured-owner"
    assert row["record_kind"] == "derived"


def test_list_active_handoffs_uses_utc_epoch_for_expiry_and_since(db_conn, mock_embedding):
    outside_since = _add(
        db_conn,
        mock_embedding,
        "created before since in offset timezone",
        created_at="2026-10-02T21:59:59+11:00",
        namespace="handoff",
        record_kind="handoff",
        expiry_at="2026-10-02T23:30:00+11:00",
    )
    active = _add(
        db_conn,
        mock_embedding,
        "active offset expiry",
        created_at="2026-10-02T22:00:00+11:00",
        namespace="handoff",
        record_kind="handoff",
        expiry_at="2026-10-02T23:00:01+11:00",
    )
    exact_expiry = _add(
        db_conn,
        mock_embedding,
        "expires exactly now offset",
        created_at="2026-10-02T22:30:00+11:00",
        namespace="handoff",
        record_kind="handoff",
        expiry_at="2026-10-02T23:00:00+11:00",
    )

    rows = db.list_active_handoffs(
        db_conn,
        agent_id="agent-a",
        since="2026-10-02T11:00:00+00:00",
        now=FROZEN_NOW,
    )

    assert [row["id"] for row in rows] == [active]
    assert outside_since not in {row["id"] for row in rows}
    assert exact_expiry not in {row["id"] for row in rows}


def test_list_active_handoffs_fails_closed_for_bad_timestamps_and_ineligible_rows(
    db_conn,
    mock_embedding,
):
    keep = _add(
        db_conn,
        mock_embedding,
        "eligible handoff",
        created_at=FROZEN_NOW.isoformat(),
        namespace="handoff",
        record_kind="handoff",
        expiry_at=(FROZEN_NOW + timedelta(hours=1)).isoformat(),
    )
    bad_expiry = _add(
        db_conn,
        mock_embedding,
        "malformed expiry",
        created_at=FROZEN_NOW.isoformat(),
        namespace="handoff",
        record_kind="handoff",
        expiry_at="not-a-timestamp",
    )
    naive_expiry = _add(
        db_conn,
        mock_embedding,
        "naive expiry",
        created_at=FROZEN_NOW.isoformat(),
        namespace="handoff",
        record_kind="handoff",
        expiry_at="2026-10-02T13:00:00",
    )
    bad_created = _add(
        db_conn,
        mock_embedding,
        "malformed created",
        created_at="not-a-timestamp",
        namespace="handoff",
        record_kind="handoff",
        expiry_at=(FROZEN_NOW + timedelta(hours=1)).isoformat(),
    )
    naive_created = _add(
        db_conn,
        mock_embedding,
        "naive created",
        created_at="2026-10-02T11:30:00",
        namespace="handoff",
        record_kind="handoff",
        expiry_at=(FROZEN_NOW + timedelta(hours=1)).isoformat(),
    )
    terminal = _add(
        db_conn,
        mock_embedding,
        "terminal handoff",
        created_at=FROZEN_NOW.isoformat(),
        namespace="handoff",
        record_kind="handoff",
        expiry_at=(FROZEN_NOW + timedelta(hours=1)).isoformat(),
        project_status="completed",
    )
    archived = _add(
        db_conn,
        mock_embedding,
        "archived handoff",
        created_at=FROZEN_NOW.isoformat(),
        namespace="handoff",
        record_kind="handoff",
        expiry_at=(FROZEN_NOW + timedelta(hours=1)).isoformat(),
    )
    db_conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (archived,))
    db_conn.commit()

    rows = db.list_active_handoffs(db_conn, agent_id="agent-a", now=FROZEN_NOW)

    assert [row["id"] for row in rows] == [keep]
    excluded = {bad_expiry, naive_expiry, bad_created, naive_created, terminal, archived}
    assert not ({row["id"] for row in rows} & excluded)


def test_agent_sync_uses_active_handoff_epoch_filtering(db_conn, mock_embedding, monkeypatch):
    keep = _add(
        db_conn,
        mock_embedding,
        "sync active offset handoff",
        created_at=FROZEN_NOW.isoformat(),
        namespace="handoff",
        record_kind="handoff",
        expiry_at="2026-10-02T23:00:01+11:00",
    )
    _add(
        db_conn,
        mock_embedding,
        "sync expired offset handoff",
        created_at=FROZEN_NOW.isoformat(),
        namespace="handoff",
        record_kind="handoff",
        expiry_at="2026-10-02T23:00:00+11:00",
    )
    import memory_v3.multi_agent as multi_agent
    import memory_v3.server as server

    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(
        multi_agent.db,
        "datetime",
        type("FrozenDateTime", (datetime,), {"now": classmethod(lambda cls, tz=None: FROZEN_NOW)}),
    )

    result = json.loads(server.agent_sync("agent-a"))

    assert result["deprecated"] is True
    assert result["mode"] == "active_handoffs"
    assert result["compatibility_mode"] is False
    assert [row["id"] for row in result["handoffs"]] == [keep]
    assert "entries" not in result


def test_agent_sync_never_uses_legacy_changelog_even_when_flag_is_set(db_conn, monkeypatch):
    monkeypatch.setenv("MEMORY_V3_LEGACY_AGENT_SYNC_ENABLED", "true")
    reset_config()
    import memory_v3.server as server

    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("changelog opened")),
    )

    result = json.loads(server.agent_sync("agent-a"))

    assert result["mode"] == "active_handoffs"
    assert result["compatibility_mode"] is False
    assert result["handoffs"] == []
    assert "entries" not in result


def test_agent_sync_failure_is_sanitized(db_conn, monkeypatch):
    import memory_v3.multi_agent as multi_agent
    import memory_v3.server as server

    monkeypatch.setattr(server, "_conn", db_conn)
    monkeypatch.setattr(
        multi_agent,
        "sync_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SECRET_SENTINEL /home/base/private.db SELECT password")
        ),
    )

    raw = server.agent_sync("agent-a")
    result = json.loads(raw)

    assert result["error"] == "Agent synchronization failed"
    assert result["mode"] == "active_handoffs"
    assert result["compatibility_mode"] is False
    assert "SECRET_SENTINEL" not in raw
    assert "/home/base" not in raw
    assert "password" not in raw
