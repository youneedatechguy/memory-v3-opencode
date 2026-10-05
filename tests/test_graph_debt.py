import builtins
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

TEST_EMBED_DIM = 768
BLOCKED_PREFIXES = (
    "memory_v3.graphs",
    "memory_v3.build_kg",
    "memory_v3.scoring.surprise",
    "memory_v3.providers",
    "networkx",
    "ollama",
    "anthropic",
    "openai",
)

ACTIVE_MODULES = (
    "memory_v3.retrieval",
    "memory_v3.server",
    "memory_v3.indexing",
    "memory_v3.lifecycle.extraction",
    "memory_v3.lifecycle.action_decider",
    "memory_v3.lifecycle.sensory",
    "memory_v3.retrieval.stage1_coarse",
    "memory_v3.retrieval.stage2_fine",
    "memory_v3.retrieval.stage3_organize",
)

STRICT_BLOCKED_PREFIXES = (
    "memory_v3.graphs",
    "memory_v3.build_kg",
    "memory_v3.scoring.surprise",
    "memory_v3.providers",
    "networkx",
    "ollama",
    "anthropic",
    "openai",
)

# The active extraction path legitimately imports the lazy provider factory
# (memory_v3.providers) for get_llm, so it is not blocked here. The guard still
# blocks centroid surprise and the ollama package.
EXTRACTION_BLOCKED_PREFIXES = (
    "memory_v3.graphs",
    "memory_v3.build_kg",
    "memory_v3.scoring.surprise",
    "networkx",
    "ollama",
    "anthropic",
    "openai",
)


class CountingEmbedder:
    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(TEST_EMBED_DIM)]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    from memory_v3.config import reset_config

    reset_config()
    server = sys.modules.get("memory_v3.server")
    if server is not None:
        if server._conn is not None:
            server._conn.close()
        server._conn = None
    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-a")
    yield
    server = sys.modules.get("memory_v3.server")
    if server is not None:
        if server._conn is not None:
            server._conn.close()
        server._conn = None
    reset_config()


@pytest.fixture
def blocked_imports(monkeypatch):
    yield from _install_blocked_imports(monkeypatch, BLOCKED_PREFIXES)


@pytest.fixture
def strict_blocked_imports(monkeypatch):
    yield from _install_blocked_imports(monkeypatch, STRICT_BLOCKED_PREFIXES)


@pytest.fixture
def extraction_blocked_imports(monkeypatch):
    yield from _install_blocked_imports(monkeypatch, EXTRACTION_BLOCKED_PREFIXES)


def _install_blocked_imports(monkeypatch, blocked_prefixes):
    original_import = builtins.__import__
    attempted: list[str] = []
    saved_modules, saved_root_attrs = _snapshot_root_package()

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        package = globals.get("__package__") if globals else None
        absolute = importlib.util.resolve_name(name, package) if level and package else name
        if absolute in blocked_prefixes or any(
            absolute.startswith(prefix + ".") for prefix in blocked_prefixes
        ):
            attempted.append(absolute)
            raise AssertionError(f"blocked import executed: {absolute}")
        return original_import(name, globals, locals, fromlist, level)

    guarded_modules = (*ACTIVE_MODULES, *blocked_prefixes)

    def should_evict(name: str) -> bool:
        return (
            name == "memory_v3"
            or name in guarded_modules
            or any(name.startswith(prefix + ".") for prefix in guarded_modules)
        )

    for name in list(sys.modules):
        if should_evict(name):
            sys.modules.pop(name, None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    try:
        yield attempted
    finally:
        _restore_root_package(saved_modules, saved_root_attrs)


def _snapshot_root_package() -> tuple[dict[str, object], dict[str, object] | None]:
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "memory_v3" or name.startswith("memory_v3.")
    }
    root = sys.modules.get("memory_v3")
    saved_root_attrs = dict(vars(root)) if root is not None else None
    return saved_modules, saved_root_attrs


def _restore_root_package(
    saved_modules: dict[str, object],
    saved_root_attrs: dict[str, object] | None,
) -> None:
    for name in list(sys.modules):
        if name == "memory_v3" or name.startswith("memory_v3."):
            sys.modules.pop(name, None)
    sys.modules.update(saved_modules)
    root = sys.modules.get("memory_v3")
    if root is not None and saved_root_attrs is not None:
        vars(root).clear()
        vars(root).update(saved_root_attrs)
    importlib.invalidate_caches()


def fresh_import(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def configure(monkeypatch, tmp_path, vault: Path) -> None:
    from memory_v3.config import reset_config

    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_VAULT", str(vault))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", "test-embed")
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "test-provider")
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", str(TEST_EMBED_DIM))
    monkeypatch.setenv("MEMORY_V3_INDEXING_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-a")
    reset_config()


def graph_dir_files(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(item.relative_to(path)) for item in path.rglob("*")}


def payload(raw: str) -> dict:
    return json.loads(raw)


def test_active_reindex_does_not_import_graph_modules_or_create_graph_artifacts(
    monkeypatch,
    tmp_path,
    blocked_imports,
):
    server = fresh_import("memory_v3.server")

    vault = tmp_path / "vault"
    write(vault / "docs/note.md", "graph inactive indexing token\n")
    configure(monkeypatch, tmp_path, vault)
    graph_dir = tmp_path / "graphs"
    monkeypatch.setattr(server, "_get_embedder", lambda: CountingEmbedder())

    result = payload(server.reindex())

    assert result == {"indexed": 1, "skipped": 0, "errors": 0, "force": False}
    assert blocked_imports == []
    assert not graph_dir.exists()


def test_retrieval_and_server_imports_stay_lean_with_fallback_and_graph_stubs(
    monkeypatch,
    strict_blocked_imports,
):
    root = fresh_import("memory_v3")
    assert root.get_version_info()["name"] == "memory-v3"

    retrieval = fresh_import("memory_v3.retrieval")
    stage1 = fresh_import("memory_v3.retrieval.stage1_coarse")
    stage2 = fresh_import("memory_v3.retrieval.stage2_fine")
    stage3 = fresh_import("memory_v3.retrieval.stage3_organize")
    indexing = fresh_import("memory_v3.indexing")
    server = fresh_import("memory_v3.server")
    db_module = importlib.import_module("memory_v3.db")

    assert retrieval.stage1_coarse is stage1
    assert retrieval.stage2_fine is stage2
    assert retrieval.stage3_organize is stage3
    assert indexing.DEFAULT_TARGET_CHARS == 3500

    def hybrid_search(conn, query, embedding, **kwargs):
        assert query == "needle"
        assert embedding == [0.1] * TEST_EMBED_DIM
        assert kwargs["limit"] == 3
        assert kwargs["retrieval_filter"] == "filter-sentinel"
        return [
            {"id": 3, "content": "third", "hybrid_score": 0.5, "bm25_score": 0.1},
            {"id": 1, "content": "first", "hybrid_score": 0.5, "bm25_score": 0.2},
            {"id": 2, "content": "second", "hybrid_score": 0.6, "bm25_score": 0.0},
        ]

    monkeypatch.setattr(db_module, "hybrid_search", hybrid_search)
    fallback = retrieval._fallback_search(
        object(),
        "needle",
        [0.1] * TEST_EMBED_DIM,
        3,
        None,
        "filter-sentinel",
    )

    assert [row["id"] for row in fallback] == [2, 1, 3]
    assert [row["final_score"] for row in fallback] == [0.6, 0.5, 0.5]

    graph_result = payload(server.rebuild_graph_debt())
    assert graph_result["code"] == "graph_lifecycle_disabled"
    hierarchy = fresh_import("memory_v3.scoring.hierarchy")
    assert hierarchy.classify_governance_layer("fact") == 3
    assert strict_blocked_imports == []
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in STRICT_BLOCKED_PREFIXES
    )


def test_extraction_path_runs_without_loading_surprise_or_ollama(
    db_conn,
    monkeypatch,
    extraction_blocked_imports,
):
    monkeypatch.setenv("MEMORY_V3_ENABLE_SENSORY", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_SURPRISE", "true")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    extraction = fresh_import("memory_v3.lifecycle.extraction")

    class StubLLM:
        def chat(self, prompt, max_tokens=2048):
            return "- deterministic extraction fact exercised under guard"

    monkeypatch.setattr(extraction, "get_llm", lambda: StubLLM())
    monkeypatch.setattr(extraction, "embed_with_cache", lambda text: [0.5] * cfg.embed_dim)

    result = extraction.process_conversation(
        db_conn,
        text="deterministic extraction fact exercised under guard",
        source="guard-test",
        config=cfg,
        add_only=True,
    )

    assert result["added"] == 1
    assert extraction_blocked_imports == []
    assert "memory_v3.scoring.surprise" not in sys.modules
    assert "ollama" not in sys.modules
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in EXTRACTION_BLOCKED_PREFIXES
    )


def test_stage1_uses_only_vector_and_fts_paths(monkeypatch, blocked_imports):
    stage1_coarse = fresh_import("memory_v3.retrieval.stage1_coarse")

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        stage1_coarse,
        "_get_vector_candidate_ids",
        lambda conn, embedding, limit=200, retrieval_filter=None: (
            calls.append(("vector", retrieval_filter)) or [1, 2]
        ),
    )
    monkeypatch.setattr(
        stage1_coarse,
        "_get_fts_candidate_ids",
        lambda conn, query, limit=200, retrieval_filter=None: (
            calls.append(("fts", retrieval_filter)) or [2, 3]
        ),
    )

    result = stage1_coarse.get_candidates(
        object(),
        "needle",
        [0.1] * TEST_EMBED_DIM,
        type("Cfg", (), {"stage1_max_candidates": 10})(),
        retrieval_filter="filter-sentinel",
    )

    assert result == [1, 2, 3]
    assert calls == [("vector", "filter-sentinel"), ("fts", "filter-sentinel")]
    assert not hasattr(stage1_coarse, "_get_community_candidate_ids")
    assert blocked_imports == []


def test_stage3_filters_only_input_candidates_and_never_expands_links_or_neighbors(
    db_conn,
    mock_embedding,
    monkeypatch,
    blocked_imports,
):
    from memory_v3 import db
    from memory_v3.retrieval.filters import RetrievalFilter

    stage3_organize = fresh_import("memory_v3.retrieval.stage3_organize")

    allowed = db.add_memory(
        db_conn,
        content="allowed stage3 token",
        embedding=mock_embedding("allowed stage3 token"),
        owner="owner-a",
        namespace="docs",
        source_file="docs/a.md",
        source_hash="a",
    )
    denied = db.add_memory(
        db_conn,
        content="denied stage3 token",
        embedding=mock_embedding("denied stage3 token"),
        owner="other-owner",
        namespace="docs",
        source_file="docs/b.md",
        source_hash="b",
    )
    rows = [
        dict(db_conn.execute("SELECT * FROM memories WHERE id = ?", (allowed,)).fetchone()),
        dict(db_conn.execute("SELECT * FROM memories WHERE id = ?", (denied,)).fetchone()),
    ]
    rows[0]["linked_memories"] = json.dumps([{"memory_id": denied, "similarity": 1.0}])
    rows[0]["related_nodes"] = [{"memory_id": denied, "node": "denied"}]
    monkeypatch.setattr(
        "memory_v3.retrieval.filters.filter_eligible_ids",
        lambda conn, ids, retrieval_filter=None: {allowed},
    )

    result = stage3_organize.organize(
        db_conn,
        rows,
        "stage3",
        type("Cfg", (), {"enable_structure_tags": False})(),
        RetrievalFilter(owner="owner-a"),
    )

    assert [row["id"] for row in result] == [allowed]
    assert result[0]["related_nodes"] == []
    assert result[0]["see_also"] == []
    assert result[0]["linked_memories"] == []
    assert "graph_rebuild_degraded" not in result[0]

    unfiltered = stage3_organize.organize(
        db_conn,
        [
            {
                "id": allowed,
                "content": "allowed stage3 token",
                "linked_memories": json.dumps([{"memory_id": denied, "similarity": 1.0}]),
            }
        ],
        "stage3",
        type("Cfg", (), {"enable_structure_tags": False})(),
    )
    assert unfiltered[0]["linked_memories"] == []
    assert blocked_imports == []


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (lambda server: server.graph_search("needle"), "graph_retrieval_disabled"),
        (lambda server: server.graph_stats_tool(), "graph_retrieval_disabled"),
        (lambda server: server.get_causal_chain("needle"), "graph_retrieval_disabled"),
        (lambda server: server.get_timeline("needle"), "graph_retrieval_disabled"),
        (lambda server: server.link_memories(1, 2), "graph_lifecycle_disabled"),
        (lambda server: server.rebuild_graph_debt(), "graph_lifecycle_disabled"),
        (lambda server: server.sleep_consolidation(), "graph_lifecycle_disabled"),
    ],
)
def test_server_graph_tools_fail_closed_before_config_db_provider_or_filesystem(
    monkeypatch,
    tmp_path,
    blocked_imports,
    call,
    code,
):
    server = fresh_import("memory_v3.server")

    graph_dir = tmp_path / "graphs"
    monkeypatch.setattr(server, "get_config", lambda: pytest.fail("config must not be read"))
    monkeypatch.setattr(server, "_get_conn", lambda: pytest.fail("db must not be opened"))
    monkeypatch.setattr(server, "_get_embedder", lambda: pytest.fail("provider must not be loaded"))

    result = payload(call(server))

    assert result["code"] == code
    assert result["deprecated"] is True
    assert result.get("data", []) == []
    assert blocked_imports == []
    assert not graph_dir.exists()


def test_indexing_graph_debt_stub_does_not_touch_db_provider_or_graph_dir(
    tmp_path,
    blocked_imports,
):
    indexing = fresh_import("memory_v3.indexing")

    class ForbiddenConn:
        def execute(self, *args, **kwargs):
            raise AssertionError("db must not be queried")

        def commit(self):
            raise AssertionError("db must not be committed")

    graph_dir = tmp_path / "graphs"

    result = indexing.rebuild_graph_debt(
        ForbiddenConn(),
        graph_dir=graph_dir,
        extraction_fn=lambda content: pytest.fail("provider must not be called"),
        publish_hook=lambda *args: pytest.fail("filesystem publish must not run"),
    )

    assert result == {
        "status": "disabled",
        "deprecated": True,
        "code": "graph_lifecycle_disabled",
        "error": "Legacy graph rebuild is disabled; active indexing is graph-free.",
        "data": [],
    }
    assert blocked_imports == []
    assert graph_dir_files(graph_dir) == set()
