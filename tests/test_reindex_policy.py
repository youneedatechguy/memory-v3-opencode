import builtins
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from memory_v3.config import reset_config
from memory_v3.retrieval.filters import RetrievalFilter

TEST_OWNER = "owner-a"


class SpyEmbedder:
    def __init__(self):
        self.inputs: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        digest = hashlib.sha256(text.encode()).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(768)]


@pytest.fixture(autouse=True)
def server_state(monkeypatch, tmp_path, request):
    if request.node.name == "test_importing_server_does_not_open_db_or_create_provider":
        yield
        reset_config()
        return

    import memory_v3.server as server

    reset_config()
    if server._conn is not None:
        server._conn.close()
    server._conn = None
    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    yield
    if server._conn is not None:
        server._conn.close()
    server._conn = None
    reset_config()


def write(path: Path, text: str = "content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def project_text(status: str, updated: str) -> str:
    return f"# Project\n\nStatus: {status}\n\nLast updated: {updated}\n\nBody.\n"


def configure(monkeypatch, tmp_path, vault: Path, *, indexing_enabled: bool = True) -> None:
    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_VAULT", str(vault))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("MEMORY_V3_INDEXING_ENABLED", "true" if indexing_enabled else "false")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-a")
    reset_config()


def make_vault(vault: Path) -> dict[str, str]:
    included = {
        "AGENTS.md": "workspace guidance\n",
        "business/pricing.md": "business pricing\n",
        "docs/guide.md": "docs guide\n",
        "projects/active/CURRENT.md": project_text("active", "2026-09-30"),
        "research/current.md": "# Research\n\nStatus: active\n",
    }
    excluded = {
        "research/archived.md": "# Research\n\nStatus: archived\n",
        "shared/private.md": "shared private\n",
        "inbox/raw.md": "raw inbox\n",
        "archive/old.md": "old root archive\n",
        ".hidden/secret.md": "hidden\n",
        "business/service-log.md": "business log\n",
        "projects/stale/CURRENT.md": project_text("active", "2026-01-01"),
        "projects/completed/CURRENT.md": project_text("completed", "2026-09-30"),
    }
    for relative, text in included.items() | excluded.items():
        write(vault / relative, text)
    return included


def test_importing_server_does_not_open_db_or_create_provider(monkeypatch, tmp_path):
    import memory_v3
    import memory_v3.db as memory_db

    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_VAULT", str(tmp_path / "vault"))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")
    monkeypatch.setenv("MEMORY_V3_INDEXING_ENABLED", "false")
    reset_config()

    sys.modules.pop("memory_v3.server", None)
    monkeypatch.setattr(
        memory_db,
        "init_db",
        lambda *args, **kwargs: pytest.fail("DB should not be opened on import"),
    )
    monkeypatch.setattr(
        memory_v3,
        "get_embedder",
        lambda *args, **kwargs: pytest.fail("embedder should not be created on import"),
    )

    server = importlib.import_module("memory_v3.server")

    assert server._conn is None


def test_reindex_kill_switch_blocks_before_db_provider_manifest_or_path_access(
    monkeypatch, tmp_path
):
    import memory_v3.server as server
    import memory_v3.source_policy as source_policy

    vault = tmp_path / "vault"
    write(vault / "AGENTS.md", "workspace guidance\n")
    configure(monkeypatch, tmp_path, vault, indexing_enabled=False)

    monkeypatch.setattr(source_policy, "build_source_manifest", lambda *a, **k: pytest.fail())
    monkeypatch.setattr(server, "_get_conn", lambda: pytest.fail("DB should not be opened"))
    monkeypatch.setattr(
        server, "get_embedder", lambda: pytest.fail("provider should not be opened")
    )
    monkeypatch.setattr(
        os.path, "isdir", lambda path: pytest.fail("filesystem should not be touched")
    )

    result = json.loads(server.reindex())

    assert result == {
        "error": "Indexing is disabled by MEMORY_V3_INDEXING_ENABLED",
        "indexed": 0,
        "skipped": 0,
        "errors": 1,
        "force": False,
    }


def test_reindex_uses_source_policy_and_writes_v4_metadata(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    included = make_vault(vault)
    configure(monkeypatch, tmp_path, vault)
    spy = SpyEmbedder()
    monkeypatch.setattr(server, "get_embedder", lambda: spy)

    result = json.loads(server.reindex())
    rows = [
        dict(row)
        for row in server._conn.execute(
            """
            SELECT source_file, owner, namespace, sensitivity, record_kind, source_id,
                   chunk_id, chunk_index, chunk_count, source_hash, source_updated_at,
                   project_status, project_updated_at, content
            FROM memories
            ORDER BY source_file
            """
        ).fetchall()
    ]

    assert result == {"indexed": len(included), "skipped": 0, "errors": 0, "force": False}
    assert set(row["source_file"] for row in rows) == set(included)
    assert len(spy.inputs) == len(included)
    assert all(any(text in embedded for embedded in spy.inputs) for text in included.values())
    assert all("shared private" not in embedded for embedded in spy.inputs)
    assert (
        memory_db.keyword_search(
            server._conn, "archived", retrieval_filter=RetrievalFilter(owner=TEST_OWNER)
        )
        == []
    )

    by_path = {row["source_file"]: row for row in rows}
    assert by_path["AGENTS.md"]["namespace"] == "workspace"
    assert by_path["business/pricing.md"]["namespace"] == "business"
    assert by_path["docs/guide.md"]["namespace"] == "docs"
    assert by_path["research/current.md"]["namespace"] == "research"
    assert by_path["projects/active/CURRENT.md"]["namespace"] == "project:active"
    assert by_path["projects/active/CURRENT.md"]["project_status"] == "active"
    assert by_path["projects/active/CURRENT.md"]["project_updated_at"] == "2026-09-30"
    for relative, row in by_path.items():
        assert row["owner"] == "owner-a"
        assert row["sensitivity"] == "normal"
        assert row["record_kind"] == "derived"
        assert row["chunk_index"] == 0
        assert row["chunk_count"] == 1
        assert row["source_hash"] == hashlib.sha256(included[relative].encode()).hexdigest()
        assert row["source_updated_at"]
        assert row["source_id"].startswith("src:")
        assert row["chunk_id"].startswith("chk:")
        assert row["content"].startswith(f"Source: {relative}\nLines: 1-")


def test_reindex_unchanged_runs_keep_stable_metadata_and_no_count_drift(monkeypatch, tmp_path):
    import memory_v3.server as server

    vault = tmp_path / "vault"
    included = make_vault(vault)
    configure(monkeypatch, tmp_path, vault)
    spy = SpyEmbedder()
    monkeypatch.setattr(server, "get_embedder", lambda: spy)

    first = json.loads(server.reindex())
    first_rows = [
        dict(row)
        for row in server._conn.execute(
            """
            SELECT source_file, source_id, chunk_id, chunk_index, chunk_count,
                   source_hash, source_updated_at, owner, namespace, record_kind
            FROM memories ORDER BY source_file
            """
        ).fetchall()
    ]
    spy.inputs.clear()
    second = json.loads(server.reindex())
    second_rows = [
        dict(row)
        for row in server._conn.execute(
            """
            SELECT source_file, source_id, chunk_id, chunk_index, chunk_count,
                   source_hash, source_updated_at, owner, namespace, record_kind
            FROM memories ORDER BY source_file
            """
        ).fetchall()
    ]

    assert first == {"indexed": len(included), "skipped": 0, "errors": 0, "force": False}
    assert second == {"indexed": 0, "skipped": len(included), "errors": 0, "force": False}
    assert first_rows == second_rows
    assert spy.inputs == []


def test_reindex_owner_isolation_for_same_source_path(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    from memory_v3.indexing import SourceIndexRecord, reindex_sources

    vault = tmp_path / "vault"
    write(vault / "docs/shared.md", "owner shared old\n")
    configure(monkeypatch, tmp_path, vault)
    conn = memory_db.init_db()
    embedder = SpyEmbedder()

    kwargs = {
        "embed_model": "test",
        "embed_provider": "test",
        "embed_dim": 768,
        "force": False,
    }
    reindex_sources(
        conn,
        vault,
        [SourceIndexRecord("docs/shared.md", "owner-a", "docs")],
        lambda: embedder,
        **kwargs,
    )
    reindex_sources(
        conn,
        vault,
        [SourceIndexRecord("docs/shared.md", "owner-b", "docs")],
        lambda: embedder,
        **kwargs,
    )
    write(vault / "docs/shared.md", "owner shared new\n")
    reindex_sources(
        conn,
        vault,
        [SourceIndexRecord("docs/shared.md", "owner-b", "docs")],
        lambda: embedder,
        **kwargs,
    )

    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT owner, content
            FROM memories
            WHERE source_file = 'docs/shared.md'
            ORDER BY owner
            """
        ).fetchall()
    ]

    assert len(rows) == 2
    assert rows[0]["owner"] == "owner-a"
    assert "owner shared old" in rows[0]["content"]
    assert rows[1]["owner"] == "owner-b"
    assert "owner shared new" in rows[1]["content"]


def test_owner_reindexes_when_global_hash_matches_other_owner(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    from memory_v3.indexing import SourceIndexRecord, reindex_sources

    vault = tmp_path / "vault"
    path = vault / "docs/shared.md"
    write(path, "old owner content\n")
    configure(monkeypatch, tmp_path, vault)
    conn = memory_db.init_db()
    embedder = SpyEmbedder()
    kwargs = {
        "embed_model": "test",
        "embed_provider": "test",
        "embed_dim": 768,
        "force": False,
    }

    reindex_sources(
        conn,
        vault,
        [SourceIndexRecord("docs/shared.md", "owner-a", "docs")],
        lambda: embedder,
        **kwargs,
    )
    owner_a_old = dict(
        conn.execute("SELECT id, source_hash FROM memories WHERE owner = 'owner-a'").fetchone()
    )
    write(path, "new owner content\n")
    reindex_sources(
        conn,
        vault,
        [SourceIndexRecord("docs/shared.md", "owner-b", "docs")],
        lambda: embedder,
        **kwargs,
    )
    embedder.inputs.clear()

    result = reindex_sources(
        conn,
        vault,
        [SourceIndexRecord("docs/shared.md", "owner-a", "docs")],
        lambda: embedder,
        **kwargs,
    )
    rows = {
        row["owner"]: dict(row)
        for row in conn.execute(
            """
            SELECT id, owner, content, source_hash
            FROM memories
            WHERE source_file = 'docs/shared.md'
            ORDER BY owner
            """
        ).fetchall()
    }

    assert result["indexed"] == 1
    assert result["skipped"] == 0
    assert len(embedder.inputs) == 1
    assert rows["owner-a"]["id"] != owner_a_old["id"]
    assert rows["owner-a"]["source_hash"] != owner_a_old["source_hash"]
    assert "new owner content" in rows["owner-a"]["content"]
    assert "new owner content" in rows["owner-b"]["content"]
    assert (
        memory_db.keyword_search(conn, "old", retrieval_filter=RetrievalFilter(owner="owner-a"))
        == []
    )
    assert (
        len(
            memory_db.keyword_search(conn, "new", retrieval_filter=RetrievalFilter(owner="owner-a"))
        )
        == 1
    )
    assert (
        len(
            memory_db.keyword_search(conn, "new", retrieval_filter=RetrievalFilter(owner="owner-b"))
        )
        == 1
    )
    assert conn.execute("SELECT COUNT(*) FROM memory_vec").fetchone()[0] == 2


def test_current_file_hash_with_stale_row_hash_embeds_and_replaces(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    import memory_v3.server as server

    vault = tmp_path / "vault"
    path = vault / "docs/stale.md"
    write(path, "fresh row content\n")
    configure(monkeypatch, tmp_path, vault)
    embedder = SpyEmbedder()
    monkeypatch.setattr(server, "get_embedder", lambda: embedder)
    assert json.loads(server.reindex())["errors"] == 0
    before = dict(
        server._conn.execute(
            "SELECT id, source_hash FROM memories WHERE source_file = 'docs/stale.md'"
        ).fetchone()
    )
    server._conn.execute(
        "UPDATE memories SET source_hash = 'stale-hash' WHERE source_file = 'docs/stale.md'"
    )
    server._conn.commit()
    embedder.inputs.clear()

    result = json.loads(server.reindex())
    after = dict(
        server._conn.execute(
            "SELECT id, source_hash FROM memories WHERE source_file = 'docs/stale.md'"
        ).fetchone()
    )

    assert result == {"indexed": 1, "skipped": 0, "errors": 0, "force": False}
    assert len(embedder.inputs) == 1
    assert after["id"] != before["id"]
    assert after["source_hash"] == before["source_hash"]
    assert memory_db.keyword_search(
        server._conn, "fresh", retrieval_filter=RetrievalFilter(owner=TEST_OWNER)
    )


def test_safe_relative_path_validation_rejects_escape_before_embedding(monkeypatch, tmp_path):
    import memory_v3.db as memory_db
    from memory_v3.indexing import SourceIndexRecord, reindex_sources

    vault = tmp_path / "vault"
    vault.mkdir()
    write(tmp_path / "outside.md", "outside\n")
    configure(monkeypatch, tmp_path, vault)
    conn = memory_db.init_db()
    embedder = SpyEmbedder()

    result = reindex_sources(
        conn,
        vault,
        [SourceIndexRecord("../outside.md", "owner-a", "docs")],
        lambda: embedder,
        embed_model="test",
        embed_provider="test",
        embed_dim=768,
        force=False,
    )

    assert result["indexed"] == 0
    assert result["errors"] == 1
    assert result["source_errors"][0]["stage"] == "read"
    assert embedder.inputs == []
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


def test_cli_reindex_reembed_fails_closed_before_db_provider_imports_and_preserves_counts(
    monkeypatch, tmp_path, capsys
):
    import memory_v3.cli as cli
    import memory_v3.db as memory_db

    db_path = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_V3_DB", str(db_path))
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", "invalid-provider")
    monkeypatch.setenv("MEMORY_V3_LLM_PROVIDER", "invalid-provider")
    reset_config()
    conn = memory_db.init_db()
    memory_db.add_memory(conn, "preserve me", [0.0] * 768)
    before = {
        "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
        "memory_vec": conn.execute("SELECT COUNT(*) FROM memory_vec").fetchone()[0],
    }
    original_import = builtins.__import__
    imported: list[str] = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked = (
            "memory_v3.server",
            "memory_v3.db",
            "memory_v3.providers",
            "memory_v3.embeddings",
            "ollama",
            "anthropic",
            "openai",
            "sentence_transformers",
        )
        if name in blocked or any(name.startswith(item + ".") for item in blocked):
            imported.append(name)
            raise AssertionError(f"blocked import executed: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["memory-v3", "reindex", "--reembed"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    after = {
        "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
        "memory_vec": conn.execute("SELECT COUNT(*) FROM memory_vec").fetchone()[0],
    }
    output = json.loads(capsys.readouterr().out)
    conn.close()

    assert exc.value.code == 1
    assert imported == []
    assert after == before == {"memories": 1, "memory_vec": 1}
    assert output == {
        "error": (
            "memory-v3 reindex --reembed is disabled; use the explicit "
            "shadow migration/reindex workflow"
        ),
        "code": "reembed_disabled",
        "status": "disabled",
    }


def test_cli_help_lists_lean_commands_and_omits_graph_debt():
    import memory_v3.cli as cli

    help_text = cli.build_parser().format_help()

    assert "reindex" in help_text
    assert "handoffs" in help_text
    assert "build-kg" in help_text
    assert "graph-debt" not in help_text
    assert "graph-debt" not in cli.COMMAND_MAP
    assert "build-kg" in cli.COMMAND_MAP


def test_cli_build_kg_fails_closed_before_config_db_provider_or_server_imports(monkeypatch, capsys):
    import memory_v3.cli as cli

    original_import = builtins.__import__
    imported: list[str] = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked = (
            "memory_v3.config",
            "memory_v3.server",
            "memory_v3.db",
            "memory_v3.providers",
            "memory_v3.embeddings",
            "memory_v3.build_kg",
            "ollama",
            "anthropic",
            "openai",
            "sentence_transformers",
        )
        if name in blocked or any(name.startswith(item + ".") for item in blocked):
            imported.append(name)
            raise AssertionError(f"blocked import executed: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(sys, "argv", ["memory-v3", "build-kg", "--force"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    output = json.loads(capsys.readouterr().out)

    assert exc.value.code == 1
    assert imported == []
    assert output == {
        "error": "memory-v3 build-kg is deprecated and disabled; graph artifacts are dormant",
        "code": "build_kg_disabled",
        "status": "disabled",
    }
