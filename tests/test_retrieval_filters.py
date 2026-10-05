import builtins
from types import SimpleNamespace

import pytest

from memory_v3 import db
from memory_v3.retrieval import stage1_coarse, stage2_fine
from memory_v3.retrieval.filters import (
    RetrievalFilter,
    build_eligibility_where,
    filter_eligible_ids,
)


def _cfg():
    return SimpleNamespace(stage1_max_candidates=10, rrf_k=60)


def _add(conn, embed, content, **metadata):
    mid = db.add_memory(
        conn,
        content=content,
        embedding=embed(content),
        source_file=metadata.pop("source_file", "/vault/project/doc.md"),
        source_hash=metadata.pop("source_hash", f"hash-{content}"),
        owner=metadata.pop("owner", "alice"),
        namespace=metadata.pop("namespace", "project:alpha"),
        author=metadata.pop("author", "indexer"),
        sensitivity=metadata.pop("sensitivity", "normal"),
        record_kind=metadata.pop("record_kind", "derived"),
        source_id=metadata.pop("source_id", f"source-{content}"),
        chunk_id=metadata.pop("chunk_id", f"chunk-{content}"),
        chunk_index=metadata.pop("chunk_index", 0),
        chunk_count=metadata.pop("chunk_count", 1),
        source_updated_at=metadata.pop("source_updated_at", "2026-01-01T00:00:00+00:00"),
        project_status=metadata.pop("project_status", "active"),
        project_updated_at=metadata.pop("project_updated_at", "2026-01-02T00:00:00+00:00"),
    )
    archived = metadata.pop("archived", False)
    assert not metadata
    if archived:
        conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (mid,))
        conn.execute("DELETE FROM memory_vec WHERE id = ?", (mid,))
        conn.commit()
    return mid


def _contents(rows):
    return {row["content"] for row in rows}


def test_owner_is_mandatory_and_parameterized():
    with pytest.raises(ValueError, match="owner"):
        build_eligibility_where(RetrievalFilter())

    filt = RetrievalFilter(
        owner="alice' OR 1=1 --",
        path_prefix="/vault/literal_100%/",
        namespaces=("project:alpha",),
    )
    sql, params = build_eligibility_where(filt)

    assert "alice" not in sql
    assert "/vault/literal" not in sql
    assert params[0] == "alice' OR 1=1 --"
    assert "/vault/literal\\_100\\%/%" in params
    for removed in (
        "source_generation",
        "deleted_at",
        "historical",
        "rank_weight",
        "rank_marker",
    ):
        assert removed not in sql


def test_cross_owner_zero_result_across_retrieval_paths(db_conn, mock_embedding):
    _add(db_conn, mock_embedding, "owner secret needle", owner="alice")
    filt = RetrievalFilter(owner="bob")

    assert stage1_coarse._get_fts_candidate_ids(db_conn, "needle", 10, filt) == []
    assert (
        stage1_coarse._get_vector_candidate_ids(
            db_conn,
            mock_embedding("owner secret needle"),
            10,
            filt,
        )
        == []
    )
    assert (
        stage1_coarse.get_candidates(
            db_conn,
            "needle",
            mock_embedding("owner secret needle"),
            _cfg(),
            filt,
        )
        == []
    )
    assert (
        stage2_fine.score_candidates(
            db_conn,
            [1],
            "needle",
            mock_embedding("owner secret needle"),
            _cfg(),
            filt,
        )
        == []
    )
    assert db.keyword_search(db_conn, "needle", retrieval_filter=filt) == []
    assert (
        db.hybrid_search(
            db_conn,
            "needle",
            mock_embedding("owner secret needle"),
            retrieval_filter=filt,
        )
        == []
    )


def test_namespace_project_path_source_author_sensitivity_kind_filters(db_conn, mock_embedding):
    wanted = _add(
        db_conn,
        mock_embedding,
        "filter needle wanted",
        namespace="project:alpha",
        source_file="/vault/projects/alpha/specs/current.md",
        source_id="source-alpha",
        author="morgan",
        sensitivity="private",
        record_kind="derived",
    )
    _add(db_conn, mock_embedding, "filter needle wrong namespace", namespace="project:beta")
    _add(db_conn, mock_embedding, "filter needle wrong source", source_id="source-beta")
    _add(db_conn, mock_embedding, "filter needle wrong author", author="casey")
    _add(db_conn, mock_embedding, "filter needle wrong sensitivity", sensitivity="normal")
    _add(db_conn, mock_embedding, "filter needle wrong kind", record_kind="handoff")

    filt = RetrievalFilter(
        owner="alice",
        namespaces=("project:alpha",),
        project_slug="alpha",
        path_prefix="/vault/projects/alpha/",
        source_id="source-alpha",
        author="morgan",
        sensitivity="private",
        record_kind="derived",
    )
    assert filter_eligible_ids(db_conn, [wanted], filt) == {wanted}
    rows = db.keyword_search(db_conn, "filter", retrieval_filter=filt, limit=10)
    assert [row["id"] for row in rows] == [wanted]


def test_current_only_status_and_archive_eligibility(db_conn, mock_embedding):
    _add(db_conn, mock_embedding, "visible active needle", project_status="active")
    _add(db_conn, mock_embedding, "visible paused needle", project_status="paused")
    _add(
        db_conn,
        mock_embedding,
        "visible research needle",
        namespace="research",
        project_status=None,
    )
    _add(db_conn, mock_embedding, "completed hidden needle", project_status="completed")
    _add(db_conn, mock_embedding, "abandoned hidden needle", project_status="abandoned")
    _add(db_conn, mock_embedding, "archived status hidden needle", project_status="archived")
    _add(db_conn, mock_embedding, "superseded hidden needle", project_status="superseded")
    _add(db_conn, mock_embedding, "stale hidden needle", project_status="stale")
    _add(db_conn, mock_embedding, "archived row hidden needle", archived=True)

    rows = db.keyword_search(
        db_conn,
        "needle",
        retrieval_filter=RetrievalFilter(owner="alice"),
        limit=20,
    )
    assert _contents(rows) == {
        "visible active needle",
        "visible paused needle",
        "visible research needle",
    }


def test_allowed_project_status_filter_can_narrow_statuses(db_conn, mock_embedding):
    active = _add(db_conn, mock_embedding, "status active needle", project_status="active")
    _add(db_conn, mock_embedding, "status paused needle", project_status="paused")
    _add(db_conn, mock_embedding, "status null needle", project_status=None)

    rows = db.keyword_search(
        db_conn,
        "status",
        retrieval_filter=RetrievalFilter(owner="alice", project_statuses=("active",)),
        limit=10,
    )
    assert [row["id"] for row in rows] == [active]


def test_allowed_project_status_filter_cannot_opt_into_terminal_statuses(
    db_conn,
    mock_embedding,
):
    _add(db_conn, mock_embedding, "status completed needle", project_status="completed")

    rows = db.keyword_search(
        db_conn,
        "status",
        retrieval_filter=RetrievalFilter(owner="alice", project_statuses=("completed",)),
        limit=10,
    )
    assert rows == []


def test_ineligible_candidates_never_reach_stage2_scoring(db_conn, mock_embedding, monkeypatch):
    eligible = _add(db_conn, mock_embedding, "zzz eligible ranker token", owner="alice")
    ineligible = _add(db_conn, mock_embedding, "ranker token", owner="bob")
    seen = []

    def spy(bm25_ranks, vec_ranks, all_ids, k=60):
        seen.extend(sorted(all_ids))
        return {mid: 1.0 for mid in all_ids}

    monkeypatch.setattr(stage2_fine, "_rrf_fusion", spy)
    rows = stage2_fine.score_candidates(
        db_conn,
        [ineligible, eligible],
        "ranker token",
        mock_embedding("ranker token"),
        _cfg(),
        RetrievalFilter(owner="alice"),
    )
    assert [row["id"] for row in rows] == [eligible]
    assert seen == [eligible]


def test_stage1_vector_eligibility_before_rank_limit_avoids_raw_neighbor_starvation(
    db_conn,
    mock_embedding,
):
    query_embedding = [0.0] * 768
    for index in range(4100):
        db.add_memory(
            db_conn,
            content=f"starvation ineligible {index}",
            embedding=query_embedding,
            source_file=f"/vault/project/ineligible-{index}.md",
            source_hash=f"hash-starvation-ineligible-{index}",
            owner="bob",
            namespace="project:alpha",
            author="indexer",
            sensitivity="normal",
            record_kind="derived",
            source_id=f"source-ineligible-{index}",
            chunk_id=f"chunk-ineligible-{index}",
            chunk_index=0,
            chunk_count=1,
            source_updated_at="2026-01-01T00:00:00+00:00",
            project_status="active",
            project_updated_at="2026-01-02T00:00:00+00:00",
        )
    eligible = db.add_memory(
        db_conn,
        content="starvation eligible farther",
        embedding=[1.0] * 768,
        source_file="/vault/project/eligible.md",
        source_hash="hash-starvation-eligible",
        owner="alice",
        namespace="project:alpha",
        author="indexer",
        sensitivity="normal",
        record_kind="derived",
        source_id="source-eligible-starvation",
        chunk_id="chunk-eligible-starvation",
        chunk_index=0,
        chunk_count=1,
        source_updated_at="2026-01-01T00:00:00+00:00",
        project_status="active",
        project_updated_at="2026-01-02T00:00:00+00:00",
    )

    rows = stage1_coarse._get_vector_candidate_ids(
        db_conn,
        query_embedding,
        limit=1,
        retrieval_filter=RetrievalFilter(owner="alice"),
    )
    assert rows == [eligible]


def test_stage2_vector_eligibility_before_rank_limit_avoids_raw_neighbor_starvation(
    db_conn,
    mock_embedding,
):
    query_embedding = [0.0] * 768
    ineligible_ids = []
    for index in range(4100):
        ineligible_ids.append(
            db.add_memory(
                db_conn,
                content=f"stage2 starvation ineligible {index}",
                embedding=query_embedding,
                source_file=f"/vault/project/stage2-ineligible-{index}.md",
                source_hash=f"hash-stage2-starvation-ineligible-{index}",
                owner="bob",
                namespace="project:alpha",
                author="indexer",
                sensitivity="normal",
                record_kind="derived",
                source_id=f"source-stage2-ineligible-{index}",
                chunk_id=f"chunk-stage2-ineligible-{index}",
                chunk_index=0,
                chunk_count=1,
                source_updated_at="2026-01-01T00:00:00+00:00",
                project_status="active",
                project_updated_at="2026-01-02T00:00:00+00:00",
            )
        )
    eligible = db.add_memory(
        db_conn,
        content="stage2 starvation eligible farther",
        embedding=[1.0] * 768,
        source_file="/vault/project/stage2-eligible.md",
        source_hash="hash-stage2-starvation-eligible",
        owner="alice",
        namespace="project:alpha",
        author="indexer",
        sensitivity="normal",
        record_kind="derived",
        source_id="source-stage2-eligible-starvation",
        chunk_id="chunk-stage2-eligible-starvation",
        chunk_index=0,
        chunk_count=1,
        source_updated_at="2026-01-01T00:00:00+00:00",
        project_status="active",
        project_updated_at="2026-01-02T00:00:00+00:00",
    )

    ranks = stage2_fine._compute_vector_ranks(
        db_conn,
        query_embedding,
        {*ineligible_ids, eligible},
        RetrievalFilter(owner="alice"),
    )
    assert ranks == {eligible: 0}


def test_vector_rank_intersects_eligible_ids_before_scoring(db_conn, mock_embedding, monkeypatch):
    eligible = _add(db_conn, mock_embedding, "vector eligible", owner="alice")
    ineligible = _add(db_conn, mock_embedding, "vector ineligible", owner="bob")
    seen = []
    original_list = stage2_fine.list_eligible_ids

    def spy_list(conn, retrieval_filter):
        result = original_list(conn, retrieval_filter)
        seen.append(result)
        return result

    monkeypatch.setattr(stage2_fine, "list_eligible_ids", spy_list)
    ranks = stage2_fine._compute_vector_ranks(
        db_conn,
        mock_embedding("vector"),
        {eligible, ineligible},
        RetrievalFilter(owner="alice"),
    )

    assert seen
    assert ineligible not in seen[0]
    assert ineligible not in ranks
    assert set(ranks) <= {eligible}


def test_stage2_uses_deterministic_tie_breakers(db_conn, mock_embedding, monkeypatch):
    first = _add(db_conn, mock_embedding, "tie first")
    second = _add(db_conn, mock_embedding, "tie second")
    monkeypatch.setattr(stage2_fine, "_compute_bm25_ranks", lambda *args, **kwargs: {})
    monkeypatch.setattr(stage2_fine, "_compute_vector_ranks", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        stage2_fine, "_rrf_fusion", lambda *args, **kwargs: {second: 1.0, first: 1.0}
    )

    rows = stage2_fine.score_candidates(
        db_conn,
        [second, first],
        "tie",
        mock_embedding("tie"),
        _cfg(),
        RetrievalFilter(owner="alice"),
    )

    assert [row["id"] for row in rows] == [first, second]


def test_path_prefix_wildcard_escaping(db_conn, mock_embedding):
    wanted = _add(
        db_conn,
        mock_embedding,
        "wildcard escape needle",
        source_file="/vault/literal_100%/doc.md",
    )
    _add(
        db_conn,
        mock_embedding,
        "wildcard escape wrong underscore",
        source_file="/vault/literalX100%/doc.md",
    )
    _add(
        db_conn,
        mock_embedding,
        "wildcard escape wrong percent",
        source_file="/vault/literal_100X/doc.md",
    )

    rows = db.keyword_search(
        db_conn,
        "wildcard",
        retrieval_filter=RetrievalFilter(owner="alice", path_prefix="/vault/literal_100%/"),
        limit=20,
    )
    assert [row["id"] for row in rows] == [wanted]


def test_provenance_contains_only_v4_retrieval_fields(db_conn, mock_embedding):
    _add(db_conn, mock_embedding, "provenance needle")
    rows = db.hybrid_search(
        db_conn,
        "provenance",
        mock_embedding("provenance needle"),
        retrieval_filter=RetrievalFilter(owner="alice"),
    )
    assert rows
    required = {
        "source_file",
        "path",
        "source_path",
        "source_hash",
        "owner",
        "namespace",
        "updated_at",
        "source_id",
        "chunk_id",
        "chunk_index",
        "chunk_count",
        "source_updated_at",
        "expiry_at",
        "record_kind",
        "sensitivity",
        "author",
        "project_status",
        "project_updated_at",
    }
    removed = {
        "source_generation_id",
        "source_generation_status",
        "rank_weight",
        "rank_marker",
        "historical",
    }
    assert required <= rows[0].keys()
    assert not (removed & rows[0].keys())


def test_stage2_import_guard_against_scoring_surprise_and_openai(db_conn, mock_embedding):
    _add(db_conn, mock_embedding, "import guard needle")
    original_import = builtins.__import__
    blocked = (
        "memory_v3.scoring",
        "memory_v3.providers.openai",
        "memory_v3.scoring.surprise",
    )

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(blocked):
            raise AssertionError(f"unexpected import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = guarded_import
        rows = stage2_fine.score_candidates(
            db_conn,
            [1],
            "import guard",
            mock_embedding("import guard"),
            _cfg(),
            RetrievalFilter(owner="alice"),
        )
    finally:
        builtins.__import__ = original_import

    assert rows
