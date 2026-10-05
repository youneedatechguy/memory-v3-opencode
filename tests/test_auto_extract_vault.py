import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


class StubLLM:
    def chat(self, prompt, max_tokens=2048):
        return "- [2026-05-03] Vault auto-extract stored fact for testing"


def set_common_env(monkeypatch, db_path: Path) -> None:
    monkeypatch.setenv("MEMORY_V3_DB", str(db_path))
    monkeypatch.setenv("MEMORY_V3_ENABLE_SENSORY", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_SURPRISE", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_FAN_EFFECT", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_THREE_STAGE", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_HALUMEM", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_CONSOLIDATION", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ZETTELKASTEN", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_MULTIGRAPH", "false")


def write(path: Path, text: str = "content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def project_text(status: str, updated: str, marker: str) -> str:
    return f"# Project\n\nStatus: {status}\nLast updated: {updated}\n\n{marker}\n"


def test_auto_extract_vault_dedup(tmp_path: Path, monkeypatch):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "AGENTS.md").write_text("hello\n", encoding="utf-8")

    db_path = tmp_path / "memory.db"
    state_path = tmp_path / "auto_extract_state.json"

    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED", "true")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    import memory_v3.lifecycle.extraction as extraction

    monkeypatch.setattr(extraction, "get_llm", lambda: StubLLM())
    monkeypatch.setattr(extraction, "embed_with_cache", lambda text: [0.01] * cfg.embed_dim)

    from memory_v3.db import init_db

    conn = init_db(str(db_path))

    from memory_v3.auto_extract.runner import AutoExtractRunner

    runner = AutoExtractRunner(conn=conn, config=cfg, state_path=state_path)

    res1 = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        vault_dir=str(vault_dir),
        dry_run=False,
        force=False,
        since_days=2,
        max_vault_files=10,
        max_chars_per_file=40000,
    )
    assert res1.skipped is False
    assert res1.extracted.get("errors", 0) == 0
    row_count_1 = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE archived = 0"
    ).fetchone()[0]
    assert row_count_1 >= 1

    res2 = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        vault_dir=str(vault_dir),
        dry_run=False,
        force=False,
        since_days=2,
        max_vault_files=10,
        max_chars_per_file=40000,
    )
    assert res2.skipped is True
    assert "No changes" in res2.reason or "Collected text is empty" in res2.reason

    row_count_2 = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE archived = 0"
    ).fetchone()[0]
    assert row_count_2 == row_count_1

    conn.close()
    reset_config()


def test_vault_auto_extract_disabled_is_noop_before_collection(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED", "false")
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    import memory_v3.auto_extract.runner as runner_mod

    calls = []

    def fail_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("vault collection should not run when disabled")

    monkeypatch.setattr(runner_mod, "compute_vault_fingerprint", fail_call)
    monkeypatch.setattr(runner_mod, "collect_vault_payloads", fail_call)
    monkeypatch.setattr(runner_mod, "process_conversation", fail_call)

    from memory_v3.auto_extract.runner import AutoExtractRunner

    runner = AutoExtractRunner(config=cfg, state_path=tmp_path / "state.json")
    result = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        vault_dir=str(tmp_path / "vault"),
    )

    assert result.skipped is True
    assert result.collected_chars == 0
    assert result.extracted == {}
    assert calls == []
    assert not db_path.exists()
    reset_config()


def test_vault_auto_extract_enabled_direct_disabled_still_processes_derived_records(
    tmp_path, monkeypatch
):
    vault = tmp_path / "vault"
    vault.mkdir()
    write(vault / "business" / "pricing.md", "vault independent marker\n")

    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    import memory_v3.auto_extract.runner as runner_mod

    calls = []

    def fake_process(conn, *, text, source, memory_metadata, add_only, **kwargs):
        calls.append(
            {
                "text": text,
                "source": source,
                "record_kind": memory_metadata["record_kind"],
                "add_only": add_only,
            }
        )
        return {
            "facts_extracted": 1,
            "added": 1,
            "updated": 0,
            "deleted": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

    monkeypatch.setattr(runner_mod, "process_conversation", fake_process)

    runner = runner_mod.AutoExtractRunner(config=cfg, state_path=tmp_path / "state.json")
    result = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        vault_dir=str(vault),
        since_days=999,
    )

    assert result.skipped is False
    assert result.extracted["added"] == 1
    assert len(calls) == 1
    assert "vault independent marker" in calls[0]["text"]
    assert calls[0]["source"] == "business/pricing.md"
    assert calls[0]["record_kind"] == "derived"
    assert calls[0]["add_only"] is True
    assert db_path.exists()
    assert (tmp_path / "state.json").exists()
    reset_config()


def test_vault_text_override_direct_disabled_skips_vault_state_db_and_processing(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    import memory_v3.auto_extract.runner as runner_mod

    calls = []

    def fail_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("vault text_override should fail closed before side effects")

    monkeypatch.setattr(runner_mod, "compute_vault_fingerprint", fail_call)
    monkeypatch.setattr(runner_mod, "collect_vault_payloads", fail_call)
    monkeypatch.setattr(runner_mod, "init_db", fail_call)
    monkeypatch.setattr(runner_mod, "process_conversation", fail_call)
    monkeypatch.setattr(runner_mod.AutoExtractState, "get", fail_call)
    monkeypatch.setattr(runner_mod.AutoExtractState, "mark_done", fail_call)

    runner = runner_mod.AutoExtractRunner(config=cfg, state_path=tmp_path / "state.json")
    result = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        text_override="direct input must not extract",
        vault_dir=str(tmp_path / "vault"),
    )

    assert result.skipped is True
    assert result.reason == "Direct writes are disabled"
    assert result.collected_chars == 0
    assert result.extracted == {}
    assert calls == []
    assert not db_path.exists()
    assert not (tmp_path / "state.json").exists()
    reset_config()


def test_vault_auto_extract_uses_source_policy_manifest(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    business = vault / "business" / "pricing.md"
    project = vault / "projects" / "active" / "CURRENT.md"
    write(vault / "AGENTS.md", "root agents included marker\n")
    write(business, "business included marker\n")
    write(vault / "docs" / "guide.md", "docs included marker\n")
    write(vault / "research" / "active.md", "Status: active\nresearch included marker\n")
    write(project, project_text("active", "2026-09-30", "active project included marker"))
    write(
        vault / "projects" / "paused" / "CURRENT.md",
        project_text("paused", "2026-09-30", "paused project included marker"),
    )
    write(vault / "inbox" / "raw.md", "inbox excluded marker\n")
    write(vault / "shared" / "family.md", "shared excluded marker\n")
    write(vault / "daily" / "2026-10-01.md", "daily excluded marker\n")
    write(vault / "archive" / "old.md", "archive excluded marker\n")
    write(vault / ".hidden" / "secret.md", "hidden excluded marker\n")
    write(vault / "business" / "logs" / "debug.md", "logs excluded marker\n")
    write(
        vault / "projects" / "stale" / "CURRENT.md",
        project_text("active", "2026-06-01", "stale project excluded marker"),
    )
    write(
        vault / "projects" / "done" / "CURRENT.md",
        project_text("completed", "2026-09-30", "terminal project excluded marker"),
    )
    write(
        vault / "research" / "superseded.md",
        "Status: superseded\nsuperseded research excluded marker\n",
    )

    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-test")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    captured_prompts = []

    class CapturingLLM:
        def chat(self, prompt, max_tokens=2048):
            captured_prompts.append(prompt)
            if "business included marker" in prompt:
                return "- [2026-10-01] business pricing source fact"
            if "active project included marker" in prompt:
                return "- [2026-10-01] active project source fact"
            return "NONE"

    import memory_v3.lifecycle.extraction as extraction

    monkeypatch.setattr(extraction, "get_llm", lambda: CapturingLLM())
    monkeypatch.setattr(extraction, "embed_with_cache", lambda text: [0.02] * cfg.embed_dim)
    monkeypatch.setattr(
        extraction,
        "decide_action",
        lambda conn, fact, embedding, config=None: {
            "action": "ADD",
            "memory_id": None,
            "reason": "test",
            "confidence": 0.9,
        },
    )

    from memory_v3.auto_extract.runner import AutoExtractRunner
    from memory_v3.db import init_db
    from memory_v3.source_policy import make_chunk_id, make_source_id

    conn = init_db(str(db_path))
    runner = AutoExtractRunner(conn=conn, config=cfg, state_path=tmp_path / "state.json")

    result = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        vault_dir=str(vault),
        since_days=999,
        max_vault_files=50,
    )

    assert result.skipped is False
    prompt = "\n".join(captured_prompts)
    for marker in [
        "root agents included marker",
        "business included marker",
        "docs included marker",
        "research included marker",
        "active project included marker",
        "paused project included marker",
    ]:
        assert marker in prompt
    for marker in [
        "inbox excluded marker",
        "shared excluded marker",
        "daily excluded marker",
        "archive excluded marker",
        "hidden excluded marker",
        "logs excluded marker",
        "stale project excluded marker",
        "terminal project excluded marker",
        "superseded research excluded marker",
    ]:
        assert marker not in prompt

    rows = conn.execute(
        """
        SELECT content, source_file, owner, namespace, sensitivity, record_kind,
               source_hash, source_id, chunk_id, chunk_index, chunk_count,
               source_updated_at, project_status, project_updated_at
        FROM memories WHERE archived = 0 ORDER BY source_file
        """
    ).fetchall()
    by_source = {row["source_file"]: dict(row) for row in rows}
    assert set(by_source) == {"business/pricing.md", "projects/active/CURRENT.md"}

    business_hash = hashlib.sha256(business.read_text(encoding="utf-8").encode()).hexdigest()
    project_hash = hashlib.sha256(project.read_text(encoding="utf-8").encode()).hexdigest()
    business_source_id = make_source_id("owner-test", "business/pricing.md")
    project_source_id = make_source_id("owner-test", "projects/active/CURRENT.md")

    expected_business = {
        "owner": "owner-test",
        "namespace": "business",
        "sensitivity": "normal",
        "record_kind": "derived",
        "source_hash": business_hash,
        "source_id": business_source_id,
        "chunk_id": make_chunk_id(business_source_id, business_hash, 0),
        "chunk_index": 0,
        "chunk_count": 1,
        "project_status": None,
        "project_updated_at": None,
    }
    expected_project = {
        "owner": "owner-test",
        "namespace": "project:active",
        "sensitivity": "normal",
        "record_kind": "derived",
        "source_hash": project_hash,
        "source_id": project_source_id,
        "chunk_id": make_chunk_id(project_source_id, project_hash, 0),
        "chunk_index": 0,
        "chunk_count": 1,
        "project_status": "active",
        "project_updated_at": "2026-09-30",
    }
    for key, value in expected_business.items():
        assert by_source["business/pricing.md"][key] == value
    for key, value in expected_project.items():
        assert by_source["projects/active/CURRENT.md"][key] == value
    for source, path in {
        "business/pricing.md": business,
        "projects/active/CURRENT.md": project,
    }.items():
        expected_mtime = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        assert by_source[source]["source_updated_at"] == expected_mtime

    conn.close()
    reset_config()


def test_direct_add_memory_disabled_skips_db_and_provider(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")

    from memory_v3.config import reset_config

    reset_config()
    import memory_v3.server as server

    calls = []

    def fail_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("direct add_memory should fail closed before side effects")

    monkeypatch.setattr(server, "_get_conn", fail_call)
    monkeypatch.setattr(server, "get_embedder", fail_call)
    monkeypatch.setattr(server.db, "add_memory", fail_call)

    result = server.add_memory("do not store this")

    assert "Direct writes are disabled" in result
    assert calls == []
    reset_config()


def test_direct_add_memory_is_insert_only_without_auto_link(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ZETTELKASTEN", "true")
    monkeypatch.setenv("MEMORY_V3_GRAPH_MUTATION_ENABLED", "true")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    from memory_v3 import db
    from memory_v3.db import init_db

    conn = init_db(str(db_path))
    existing_id = db.add_memory(
        conn,
        content="Existing row must keep links unchanged",
        embedding=[0.3] * cfg.embed_dim,
        content_type="fact",
    )

    import memory_v3.graphs.zettelkasten as zettelkasten
    import memory_v3.server as server

    class DirectEmbedder:
        def embed(self, text):
            return [0.3] * cfg.embed_dim

    def fail_auto_link(*args, **kwargs):
        raise AssertionError("direct add_memory must not auto-link")

    monkeypatch.setattr(server, "_conn", conn)
    monkeypatch.setattr(server, "get_embedder", lambda: DirectEmbedder())
    monkeypatch.setattr(zettelkasten, "auto_link", fail_auto_link)

    result = json.loads(server.add_memory("New handoff row"))

    assert result["memory_id"] != existing_id
    old = conn.execute(
        "SELECT content, linked_memories FROM memories WHERE id = ?",
        (existing_id,),
    ).fetchone()
    assert old["content"] == "Existing row must keep links unchanged"
    assert json.loads(old["linked_memories"]) == []

    server._conn = None
    conn.close()
    reset_config()


def test_direct_extract_disabled_skips_db_provider_and_insert(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")

    from memory_v3.config import reset_config

    reset_config()
    import memory_v3.server as server

    calls = []

    def fail_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("direct extract should fail closed before side effects")

    monkeypatch.setattr(server, "_get_conn", fail_call)
    monkeypatch.setattr(server, "get_embedder", fail_call)
    monkeypatch.setattr(server.db, "add_memory", fail_call)

    result = server.extract_from_conversation("do not extract this")

    assert "Direct writes are disabled" in result
    assert calls == []
    reset_config()


def test_direct_extract_is_add_only_expiring_handoff(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ZETTELKASTEN", "true")
    monkeypatch.setenv("MEMORY_V3_GRAPH_MUTATION_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-direct")
    monkeypatch.setenv("MEMORY_V3_HANDOFF_TTL_DAYS", "7")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    from memory_v3 import db
    from memory_v3.db import init_db

    conn = init_db(str(db_path))
    existing_id = db.add_memory(
        conn,
        content="Existing durable record must not be updated or deleted",
        embedding=[0.5] * cfg.embed_dim,
        content_type="fact",
        source_file="business/durable.md",
        author="indexer",
        owner="owner-direct",
        namespace="business",
        sensitivity="normal",
        record_kind="derived",
    )

    import memory_v3.graphs.zettelkasten as zettelkasten
    import memory_v3.lifecycle.extraction as extraction
    import memory_v3.server as server

    class DirectLLM:
        def chat(self, prompt, max_tokens=2048):
            return "- Existing durable record must be replaced by a handoff"

    def fail_decide(*args, **kwargs):
        raise AssertionError("direct extraction must not update/delete via decide_action")

    def fail_auto_link(*args, **kwargs):
        raise AssertionError("direct add-only extraction must not auto-link")

    monkeypatch.setattr(extraction, "get_llm", lambda: DirectLLM())
    monkeypatch.setattr(extraction, "embed_with_cache", lambda text: [0.5] * cfg.embed_dim)
    monkeypatch.setattr(extraction, "decide_action", fail_decide)
    monkeypatch.setattr(zettelkasten, "auto_link", fail_auto_link)
    monkeypatch.setattr(server, "_conn", conn)

    result = json.loads(server.extract_from_conversation("replace durable record"))

    assert result["added"] == 1
    assert result["updated"] == 0
    assert result["deleted"] == 0

    old = conn.execute(
        "SELECT content, archived, linked_memories FROM memories WHERE id = ?",
        (existing_id,),
    ).fetchone()
    assert old["content"] == "Existing durable record must not be updated or deleted"
    assert old["archived"] == 0
    assert json.loads(old["linked_memories"]) == []

    rows = conn.execute(
        """
        SELECT id, owner, namespace, sensitivity, record_kind, expiry_at
        FROM memories WHERE archived = 0 ORDER BY id
        """
    ).fetchall()
    assert len(rows) == 2
    handoff = dict(rows[-1])
    assert handoff["owner"] == "owner-direct"
    assert handoff["namespace"] == "handoff"
    assert handoff["sensitivity"] == "normal"
    assert handoff["record_kind"] == "handoff"
    expiry = datetime.fromisoformat(handoff["expiry_at"])
    assert expiry > datetime.now(timezone.utc) + timedelta(days=6)

    server._conn = None
    conn.close()
    reset_config()


def test_normal_extraction_graph_disabled_skips_auto_link(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_ENABLE_ZETTELKASTEN", "true")
    monkeypatch.setenv("MEMORY_V3_GRAPH_MUTATION_ENABLED", "false")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    from memory_v3.db import init_db

    conn = init_db(str(db_path))

    import memory_v3.graphs.zettelkasten as zettelkasten
    import memory_v3.lifecycle.extraction as extraction

    class NormalLLM:
        def chat(self, prompt, max_tokens=2048):
            return "- normal extraction should add without graph mutation"

    def add_decision(*args, **kwargs):
        return {
            "action": "ADD",
            "memory_id": None,
            "reason": "test",
            "confidence": 0.9,
        }

    def fail_auto_link(*args, **kwargs):
        raise AssertionError("graph-disabled normal extraction must not auto-link")

    monkeypatch.setattr(extraction, "get_llm", lambda: NormalLLM())
    monkeypatch.setattr(extraction, "embed_with_cache", lambda text: [0.6] * cfg.embed_dim)
    monkeypatch.setattr(extraction, "decide_action", add_decision)
    monkeypatch.setattr(zettelkasten, "auto_link", fail_auto_link)

    result = extraction.process_conversation(
        conn,
        text="normal extraction",
        source="conversation",
        config=cfg,
        add_only=False,
    )

    assert result["added"] == 1
    assert result["updated"] == 0
    assert result["deleted"] == 0

    row = conn.execute("SELECT linked_memories FROM memories").fetchone()
    assert json.loads(row["linked_memories"]) == []

    conn.close()
    reset_config()


def test_vault_error_does_not_advance_checkpoint_and_retries(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    write(vault / "business" / "pricing.md", "business retry marker\n")
    write(vault / "docs" / "guide.md", "docs retry marker\n")

    db_path = tmp_path / "memory.db"
    state_path = tmp_path / "state.json"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED", "true")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    import memory_v3.auto_extract.runner as runner_mod
    from memory_v3.auto_extract.runner import AutoExtractRunner

    calls: list[str] = []
    failures_left = {"business/pricing.md": 1}

    def fake_process(conn, *, text, source, **kwargs):
        calls.append(source)
        if failures_left.get(source, 0):
            failures_left[source] -= 1
            return {
                "facts_extracted": 1,
                "added": 0,
                "updated": 0,
                "deleted": 0,
                "skipped": 0,
                "errors": 1,
                "details": [{"action": "ERROR", "source_file": source}],
            }
        return {
            "facts_extracted": 1,
            "added": 1,
            "updated": 0,
            "deleted": 0,
            "skipped": 0,
            "errors": 0,
            "details": [{"action": "ADD", "source_file": source}],
        }

    monkeypatch.setattr(runner_mod, "process_conversation", fake_process)

    runner = AutoExtractRunner(config=cfg, state_path=state_path)
    first = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        vault_dir=str(vault),
        since_days=999,
    )

    assert first.extracted["errors"] == 1
    assert not state_path.exists()
    assert calls.count("business/pricing.md") == 1
    assert calls.count("docs/guide.md") == 1

    second = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        vault_dir=str(vault),
        since_days=999,
    )

    assert second.extracted["errors"] == 0
    assert state_path.exists()
    assert calls.count("business/pricing.md") == 2
    assert calls.count("docs/guide.md") == 2
    reset_config()


def test_vault_derived_extraction_is_add_only_and_preserves_other_source(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    write(vault / "business" / "pricing.md", "vault add only marker\n")

    db_path = tmp_path / "memory.db"
    set_common_env(monkeypatch, db_path)
    monkeypatch.setenv("MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ZETTELKASTEN", "true")
    monkeypatch.setenv("MEMORY_V3_GRAPH_MUTATION_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-vault")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    from memory_v3 import db
    from memory_v3.db import init_db

    conn = init_db(str(db_path))
    existing_id = db.add_memory(
        conn,
        content="Existing other source must survive vault extraction",
        embedding=[0.4] * cfg.embed_dim,
        content_type="fact",
        source_file="docs/other.md",
        author="indexer",
        owner="owner-vault",
        namespace="docs",
        sensitivity="normal",
        record_kind="derived",
    )

    import memory_v3.graphs.zettelkasten as zettelkasten
    import memory_v3.lifecycle.extraction as extraction

    class VaultLLM:
        def chat(self, prompt, max_tokens=2048):
            return "- Existing other source must survive vault extraction"

    def fail_decide(*args, **kwargs):
        raise AssertionError("vault derived extraction must be add-only")

    def fail_auto_link(*args, **kwargs):
        raise AssertionError("vault derived add-only extraction must not auto-link")

    monkeypatch.setattr(extraction, "get_llm", lambda: VaultLLM())
    monkeypatch.setattr(extraction, "embed_with_cache", lambda text: [0.4] * cfg.embed_dim)
    monkeypatch.setattr(extraction, "decide_action", fail_decide)
    monkeypatch.setattr(zettelkasten, "auto_link", fail_auto_link)

    from memory_v3.auto_extract.runner import AutoExtractRunner

    runner = AutoExtractRunner(conn=conn, config=cfg, state_path=tmp_path / "state.json")
    result = runner.run(
        trigger_type="vault",
        fingerprint="auto",
        vault_dir=str(vault),
        since_days=999,
    )

    assert result.extracted["added"] == 1
    assert result.extracted["updated"] == 0
    assert result.extracted["deleted"] == 0

    old = conn.execute(
        "SELECT content, source_file, archived, linked_memories FROM memories WHERE id = ?",
        (existing_id,),
    ).fetchone()
    assert old["content"] == "Existing other source must survive vault extraction"
    assert old["source_file"] == "docs/other.md"
    assert old["archived"] == 0
    assert json.loads(old["linked_memories"]) == []

    added = conn.execute(
        """
        SELECT source_file, owner, namespace, record_kind, expiry_at
        FROM memories WHERE id != ? AND archived = 0
        """,
        (existing_id,),
    ).fetchone()
    assert dict(added) == {
        "source_file": "business/pricing.md",
        "owner": "owner-vault",
        "namespace": "business",
        "record_kind": "derived",
        "expiry_at": None,
    }

    conn.close()
    reset_config()
