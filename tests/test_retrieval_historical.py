from types import SimpleNamespace

import pytest

from memory_v3 import db
from memory_v3.retrieval import stage2_fine
from memory_v3.retrieval.filters import RetrievalFilter


def _cfg():
    return SimpleNamespace(rrf_k=60)


def _add(conn, embed, content, **metadata):
    mid = db.add_memory(
        conn,
        content=content,
        embedding=embed(content),
        source_file=metadata.pop("source_file", "research/doc.md"),
        source_hash=metadata.pop("source_hash", f"hash-{content}"),
        owner=metadata.pop("owner", "alice"),
        namespace=metadata.pop("namespace", "research"),
        author=metadata.pop("author", "indexer"),
        sensitivity=metadata.pop("sensitivity", "normal"),
        record_kind=metadata.pop("record_kind", "derived"),
        source_id=metadata.pop("source_id", f"source-{content}"),
        chunk_id=metadata.pop("chunk_id", f"chunk-{content}"),
        chunk_index=metadata.pop("chunk_index", 0),
        chunk_count=metadata.pop("chunk_count", 1),
        source_updated_at=metadata.pop("source_updated_at", "2026-01-01T00:00:00+00:00"),
        project_status=metadata.pop("project_status", None),
        project_updated_at=metadata.pop("project_updated_at", None),
    )
    archived = metadata.pop("archived", False)
    assert not metadata
    if archived:
        conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (mid,))
        conn.execute("DELETE FROM memory_vec WHERE id = ?", (mid,))
        conn.commit()
    return mid


def _ids(rows):
    return [row["id"] for row in rows]


@pytest.mark.parametrize(
    "terminal_status", ["completed", "abandoned", "archived", "superseded", "stale"]
)
def test_terminal_research_is_absent_from_keyword_hybrid_and_stage2(
    db_conn,
    mock_embedding,
    terminal_status,
):
    current = _add(
        db_conn,
        mock_embedding,
        f"history {terminal_status} current",
        source_file="research/current.md",
        project_status=None,
    )
    terminal = _add(
        db_conn,
        mock_embedding,
        f"history {terminal_status} hidden",
        source_file=f"research/{terminal_status}.md",
        project_status=terminal_status,
    )
    filt = RetrievalFilter(owner="alice", namespaces=("research",))

    keyword_rows = db.keyword_search(db_conn, "history", retrieval_filter=filt, limit=20)
    hybrid_rows = db.hybrid_search(
        db_conn,
        "history",
        mock_embedding("history"),
        retrieval_filter=filt,
        limit=20,
    )
    stage2_rows = stage2_fine.score_candidates(
        db_conn,
        [terminal, current],
        "history",
        mock_embedding("history"),
        _cfg(),
        filt,
    )

    assert _ids(keyword_rows) == [current]
    assert _ids(hybrid_rows) == [current]
    assert _ids(stage2_rows) == [current]


def test_archived_memory_rows_are_absent_in_every_retrieval_mode(db_conn, mock_embedding):
    current = _add(
        db_conn,
        mock_embedding,
        "archived row current",
        source_file="research/current.md",
    )
    archived = _add(
        db_conn,
        mock_embedding,
        "archived row hidden",
        source_file="research/archived-row.md",
        archived=True,
    )
    filt = RetrievalFilter(owner="alice", namespaces=("research",))

    assert _ids(db.keyword_search(db_conn, "archived", retrieval_filter=filt, limit=20)) == [
        current
    ]
    assert _ids(
        db.hybrid_search(
            db_conn,
            "archived",
            mock_embedding("archived"),
            retrieval_filter=filt,
            limit=20,
        )
    ) == [current]
    assert _ids(
        stage2_fine.score_candidates(
            db_conn,
            [archived, current],
            "archived",
            mock_embedding("archived"),
            _cfg(),
            filt,
        )
    ) == [current]


def test_historical_opt_in_filter_api_is_removed():
    with pytest.raises(TypeError):
        RetrievalFilter(owner="alice", historical=True)
