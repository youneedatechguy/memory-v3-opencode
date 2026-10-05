import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def write(path: Path, text: str = "content long enough for extraction policy tests\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def project_text(status: str, updated: str) -> str:
    return f"# Project\n\nStatus: {status}\nLast updated: {updated}\n\nBody.\n"


def configure(monkeypatch, tmp_path, vault: Path) -> None:
    from memory_v3.config import reset_config

    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_VAULT", str(vault))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    reset_config()


def make_policy_vault(vault: Path) -> set[str]:
    included = {
        "AGENTS.md",
        "business/pricing.md",
        "docs/guide.md",
        "research/current.md",
        "projects/active/specs/CURRENT.md",
        "projects/paused/CURRENT.md",
    }
    write(
        vault / "AGENTS.md",
        "workspace guidance long enough for extraction and graph mutation\n",
    )
    write(
        vault / "business" / "pricing.md",
        "pricing notes long enough for extraction and graph mutation\n",
    )
    write(
        vault / "docs" / "guide.md",
        "guide notes long enough for extraction and graph mutation\n",
    )
    write(
        vault / "research" / "current.md",
        "# Research\n\nStatus: active\n\n"
        "Current notes long enough for extraction and graph mutation.\n",
    )
    write(vault / "research" / "archived.md", "# Research\n\nStatus: archived\n\nOld notes.\n")
    write(
        vault / "projects" / "active" / "specs" / "CURRENT.md",
        project_text("active", "2026-09-30"),
    )
    write(vault / "projects" / "paused" / "CURRENT.md", project_text("paused", "2026-09-30"))
    write(vault / "projects" / "stale" / "CURRENT.md", project_text("active", "2026-01-01"))
    write(vault / "projects" / "completed" / "CURRENT.md", project_text("completed", "2026-09-30"))
    write(vault / "inbox" / "raw.md", "private inbox long enough for extraction\n")
    write(vault / "daily" / "2026-10-01.md", "daily long enough for extraction\n")
    write(vault / "archive" / "old.md", "archive long enough for extraction\n")
    write(vault / "shared" / "private.md", "shared long enough for extraction\n")
    write(vault / "LifeOS" / "inactive.md", "lifeos long enough for extraction\n")
    write(vault / ".hidden" / "secret.md", "hidden long enough for extraction\n")
    write(vault / "business" / "service-log.md", "log long enough for extraction\n")
    write(vault / "business" / "logs" / "incident.md", "log long enough for extraction\n")
    return included


class FakeConn:
    def __init__(self):
        self.selects: list[str] = []
        self.writes: list[str] = []
        self.hashes: dict[str, str] = {}
        self.last_select: str | None = None
        self.commits = 0

    def execute(self, sql, params=()):
        if sql.lstrip().upper().startswith("SELECT"):
            self.selects.append(params[0])
            self.last_select = params[0]
            return self
        self.writes.append(params[0])
        if len(params) > 1:
            self.hashes[params[0]] = params[1]
        return self

    def fetchone(self):
        if self.last_select in self.hashes:
            return (self.hashes[self.last_select],)
        return None

    def commit(self):
        self.commits += 1


def assert_disabled_build_kg_result(result: dict, backend: str = "ollama") -> None:
    assert result == {
        "status": "disabled",
        "deprecated": True,
        "code": "knowledge_graph_disabled",
        "error": (
            "Legacy knowledge graph building is disabled; active memory-v3 operation is graph-free."
        ),
        "files_processed": 0,
        "files_skipped": 0,
        "files_errored": 0,
        "total_entities": 0,
        "total_relationships": 0,
        "per_layer": {},
        "backend": backend,
    }


def test_build_kg_disabled_is_noop_before_config_manifest_db_provider_or_graph(
    monkeypatch,
    tmp_path,
):
    import memory_v3.build_kg as build_kg

    vault = tmp_path / "vault"
    vault.mkdir()
    make_policy_vault(vault)

    monkeypatch.setattr(
        build_kg,
        "_extract_ollama",
        lambda *args, **kwargs: pytest.fail("provider should not be called"),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: pytest.fail("vault files should not be read"),
    )
    monkeypatch.setattr(
        Path, "rglob", lambda *args, **kwargs: pytest.fail("manifest should not be read")
    )

    result = build_kg.build_knowledge_graph(str(vault), backend="ollama")

    assert_disabled_build_kg_result(result)
    assert not (tmp_path / "graphs").exists()


def test_build_kg_disabled_does_not_import_db_policy_graph_or_provider(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    code = f"""
import importlib.abc
import json
import os
import sys

import memory_v3

blocked = {{
    "sqlite_vec",
    "ollama",
    "anthropic",
    "openai",
    "memory_v3.config",
    "memory_v3.db",
    "memory_v3.source_policy",
    "memory_v3.graphs",
    "memory_v3.providers",
}}

for module_name in list(sys.modules):
    if module_name in blocked or any(module_name.startswith(item + ".") for item in blocked):
        sys.modules.pop(module_name)
for child_name in ("config", "db", "source_policy", "graphs", "providers"):
    if hasattr(memory_v3, child_name):
        delattr(memory_v3, child_name)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or any(fullname.startswith(item + ".") for item in blocked):
            raise AssertionError(f"blocked import: {{fullname}}")
        return None

sys.meta_path.insert(0, Blocker())
os.environ["MEMORY_V3_VAULT"] = {str(vault)!r}
import memory_v3.build_kg as build_kg

result = build_kg.build_knowledge_graph({str(vault)!r}, backend="ollama")
leaked = [
    module_name
    for module_name in sys.modules
    if module_name in blocked or any(module_name.startswith(item + ".") for item in blocked)
]
if leaked:
    raise AssertionError(f"blocked module cached: {{leaked}}")
print(json.dumps(result, sort_keys=True))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert_disabled_build_kg_result(payload)


def test_build_kg_disabled_does_not_write_hashes_or_graph_artifacts(monkeypatch, tmp_path):
    import memory_v3.build_kg as build_kg

    vault = tmp_path / "vault"
    vault.mkdir()
    make_policy_vault(vault)
    conn = FakeConn()

    monkeypatch.setattr(
        build_kg,
        "_extract_ollama",
        lambda *args, **kwargs: pytest.fail("provider should not be called"),
    )

    result = build_kg.build_knowledge_graph(str(vault), conn=conn, backend="ollama")

    assert_disabled_build_kg_result(result)
    assert conn.selects == []
    assert conn.writes == []
    assert conn.hashes == {}
    assert conn.commits == 0
    assert not (tmp_path / "graphs").exists()


def test_cli_build_kg_is_stable_disabled_before_config_db_provider_or_graph(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    code = f"""
import importlib.abc
import json
import os
import sys

import memory_v3

blocked = {{
    "sqlite_vec",
    "ollama",
    "anthropic",
    "openai",
    "memory_v3.config",
    "memory_v3.db",
    "memory_v3.source_policy",
    "memory_v3.graphs",
    "memory_v3.providers",
}}

for module_name in list(sys.modules):
    if module_name in blocked or any(module_name.startswith(item + ".") for item in blocked):
        sys.modules.pop(module_name)
for child_name in ("config", "db", "source_policy", "graphs", "providers"):
    if hasattr(memory_v3, child_name):
        delattr(memory_v3, child_name)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or any(fullname.startswith(item + ".") for item in blocked):
            raise AssertionError(f"blocked import: {{fullname}}")
        return None

sys.meta_path.insert(0, Blocker())
os.environ["MEMORY_V3_DB"] = {str(tmp_path / "memory.db")!r}
os.environ["MEMORY_V3_VAULT"] = {str(vault)!r}

from memory_v3.cli import main

sys.argv = ["memory-v3", "build-kg"]
try:
    main()
except SystemExit as exc:
    leaked = [
        module_name
        for module_name in sys.modules
        if module_name in blocked or any(module_name.startswith(item + ".") for item in blocked)
    ]
    if leaked:
        raise AssertionError(f"blocked module cached: {{leaked}}")
    print(json.dumps({{"exit_code": exc.code}}))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    payload = json.loads("\n".join(lines[:-1]))
    exit_payload = json.loads(lines[-1])

    assert payload == {
        "error": "memory-v3 build-kg is deprecated and disabled; graph artifacts are dormant",
        "code": "build_kg_disabled",
        "status": "disabled",
    }
    assert exit_payload == {"exit_code": 1}
    assert not (tmp_path / "memory.db").exists()


def test_integrity_manifest_matches_central_source_policy(monkeypatch, tmp_path):
    from memory_v3.config import reset_config
    from memory_v3.security import generate_manifest
    from memory_v3.source_policy import build_source_manifest

    vault = tmp_path / "vault"
    vault.mkdir()
    make_policy_vault(vault)
    configure(monkeypatch, tmp_path, vault)

    manifest = generate_manifest(vault)
    policy_paths = [record.relative_path for record in build_source_manifest(vault)]

    assert list(manifest["files"].keys()) == policy_paths
    assert set(manifest["files"]) == set(make_policy_vault(vault))
    reset_config()


def test_integrity_check_reports_old_excluded_manifest_entries_separately(
    monkeypatch,
    tmp_path,
):
    from memory_v3.config import reset_config
    from memory_v3.security import check_integrity

    vault = tmp_path / "vault"
    vault.mkdir()
    write(vault / "docs" / "guide.md", "new approved content\n")
    write(vault / "inbox" / "raw.md", "still excluded old content\n")
    configure(monkeypatch, tmp_path, vault)
    old_hash = hashlib.sha256(b"old approved content\n").hexdigest()
    excluded_hash = hashlib.sha256((vault / "inbox" / "raw.md").read_bytes()).hexdigest()
    manifest = {
        "generated_at": "2026-10-01T00:00:00+00:00",
        "files": {
            "docs/guide.md": {"sha256": old_hash, "size": 21, "last_modified": "old"},
            "docs/missing.md": {"sha256": old_hash, "size": 21, "last_modified": "old"},
            "inbox/raw.md": {"sha256": excluded_hash, "size": 27, "last_modified": "old"},
        },
    }
    (vault / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = check_integrity(vault)

    assert result["modified"] == ["docs/guide.md"]
    assert result["missing"] == ["docs/missing.md"]
    assert result["excluded_by_policy"] == ["inbox/raw.md"]
    reset_config()


def insert_memory(conn, source_file: str, source_hash: str | None = None) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """INSERT INTO memories
        (content, source_file, created_at, updated_at, last_accessed_at, source_hash)
        VALUES (?, ?, ?, ?, ?, ?)""",
        ("memory content", source_file, now, now, now, source_hash),
    )
    conn.commit()
    return cursor.lastrowid


def verify_status(conn, source_file: str, source_hash: str | None = None) -> dict:
    from memory_v3.lifecycle.hallucination import verify_memory

    memory_id = insert_memory(conn, source_file, source_hash)
    return verify_memory(conn, memory_id)


def test_hallucination_verification_allows_approved_relative_and_absolute_sources(
    monkeypatch,
    tmp_path,
):
    from memory_v3.config import reset_config
    from memory_v3.db import init_db

    vault = tmp_path / "vault"
    vault.mkdir()
    make_policy_vault(vault)
    configure(monkeypatch, tmp_path, vault)
    conn = init_db(str(tmp_path / "memory.db"))

    rel_hash = hashlib.sha256((vault / "docs" / "guide.md").read_bytes()).hexdigest()
    abs_hash = hashlib.sha256((vault / "business" / "pricing.md").read_bytes()).hexdigest()

    rel_result = verify_status(conn, "docs/guide.md", rel_hash)
    abs_result = verify_status(conn, str(vault / "business" / "pricing.md"), abs_hash)

    assert rel_result["status"] == "verified"
    assert abs_result["status"] == "verified"
    conn.close()
    reset_config()


@pytest.mark.parametrize(
    "source_file",
    [
        "inbox/raw.md",
        "../outside.md",
        None,
        "docs/escape.md",
    ],
)
def test_hallucination_verification_rejects_excluded_traversal_outside_and_symlink(
    monkeypatch,
    tmp_path,
    source_file,
):
    from memory_v3.config import reset_config
    from memory_v3.db import init_db

    vault = tmp_path / "vault"
    vault.mkdir()
    make_policy_vault(vault)
    outside = tmp_path / "outside.md"
    write(outside, "outside long enough for extraction\n")
    (vault / "docs" / "escape.md").symlink_to(outside)
    actual_source = str(outside) if source_file is None else source_file
    configure(monkeypatch, tmp_path, vault)
    conn = init_db(str(tmp_path / "memory.db"))

    result = verify_status(conn, actual_source)

    assert result["status"] == "stale"
    assert result["reason"] == "Source path is outside approved source policy"
    conn.close()
    reset_config()
