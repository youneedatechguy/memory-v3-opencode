from dataclasses import asdict
from datetime import date
from inspect import signature
from pathlib import Path

from memory_v3.config import Config, get_config, reset_config
from memory_v3.source_policy import SourceRecord, build_source_manifest

TODAY = date(2026, 10, 1)


def write(path: Path, text: str = "content\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def project_text(status: str, updated: str) -> str:
    return f"# Project\n\nStatus: {status}\n\nLast updated: {updated}\n\nNext action.\n"


def build_fixture(vault: Path) -> None:
    write(vault / "AGENTS.md", "workspace guidance\n")
    write(vault / "README.md", "unrelated root file\n")
    write(vault / "sensitive" / "CURRENT.md", project_text("active", "2026-09-30"))
    write(vault / "business" / "pricing.md")
    write(vault / "business" / "catalog.md")
    write(vault / "business" / "changelog.md")
    write(vault / "business" / "archive" / "old.md")
    write(vault / "business" / "debug.log")
    write(vault / "business" / "service-log.md")
    write(vault / "business" / "error.log.md")
    write(vault / "business" / "logs" / "incident.md")
    write(vault / "business" / ".scratchpad.md")
    write(vault / "business" / "empty.md", "")
    write(vault / "business" / "notes.tmp")
    write(vault / "decisions" / "001.md")
    write(vault / "infrastructure" / "hosts.md")
    write(vault / "docs" / "guide.md")
    write(vault / "docs" / "archive" / "old-guide.md")
    write(
        vault / "research" / "current.md",
        "# Research\n\nStatus: active\nLast updated: 2026-09-30\n",
    )
    write(vault / "research" / "large-body.md", "# Research\nStatus: active\n" + "x" * 20000)
    write(vault / "research" / "no-status.md", "# Research\n")
    write(
        vault / "research" / "old-active.md",
        "# Research\n\nStatus: active\nLast updated: 2026-01-01\n",
    )
    write(vault / "research" / "archived.md", "# Research\n\nStatus: archived\n")
    write(vault / "research" / "superseded.md", "# Research\n\nStatus: superseded\n")
    write(vault / "research" / "archive" / "nested.md", "# Research\n")
    write(
        vault / "projects" / "active" / "specs" / "CURRENT.md",
        project_text("active", "2026-07-03"),
    )
    write(vault / "projects" / "paused" / "CURRENT.md", project_text("paused", "2026-09-30"))
    write(vault / "projects" / "stale" / "CURRENT.md", project_text("active", "2026-07-02"))
    write(vault / "projects" / "future" / "CURRENT.md", project_text("active", "2026-10-02"))
    write(vault / "projects" / "completed" / "CURRENT.md", project_text("completed", "2026-09-30"))
    write(vault / "projects" / "abandoned" / "CURRENT.md", project_text("abandoned", "2026-09-30"))
    write(vault / "projects" / "archived" / "CURRENT.md", project_text("archived", "2026-09-30"))
    write(
        vault / "projects" / "superseded" / "CURRENT.md",
        project_text("superseded", "2026-09-30"),
    )
    write(
        vault / "projects" / "malformed-date" / "CURRENT.md",
        project_text("active", "2026-99-99"),
    )
    write(vault / "projects" / "missing-date" / "CURRENT.md", "Status: active\n")
    write(vault / "projects" / "missing-status" / "CURRENT.md", "Last updated: 2026-09-30\n")
    write(
        vault / "projects" / "unknown-status" / "CURRENT.md",
        project_text("blocked", "2026-09-30"),
    )
    write(vault / "projects" / "empty" / "CURRENT.md", "")
    write(
        vault / "projects" / "active" / "archive" / "CURRENT.md",
        project_text("active", "2026-09-30"),
    )
    write(vault / "projects" / "active" / "notes.md", project_text("active", "2026-09-30"))
    write(vault / "inbox" / "raw.md")
    write(vault / "daily" / "2026-10-01.md")
    write(vault / "archive" / "root-old.md")
    write(vault / ".trash" / "deleted.md")
    write(vault / "shared" / "wife-boundary.md")
    write(vault / "LifeOS" / "CURRENT.md", project_text("active", "2026-09-30"))
    write(vault / ".hidden" / "secret.md")
    write(vault / "business" / ".hidden-dir" / "secret.md")

    outside = vault.parent / "outside"
    write(outside / "escape.md")
    (vault / "business" / "escape.md").symlink_to(outside / "escape.md")
    (vault / "docs" / "escape-dir").symlink_to(outside, target_is_directory=True)


def normal_expected(owner: str = "owner-a") -> list[SourceRecord]:
    return [
        SourceRecord("AGENTS.md", owner, "workspace"),
        SourceRecord("business/catalog.md", owner, "business"),
        SourceRecord("business/changelog.md", owner, "business"),
        SourceRecord("business/pricing.md", owner, "business"),
        SourceRecord("decisions/001.md", owner, "decisions"),
        SourceRecord("docs/guide.md", owner, "docs"),
        SourceRecord("infrastructure/hosts.md", owner, "infrastructure"),
        SourceRecord(
            "projects/active/specs/CURRENT.md",
            owner,
            "project:active",
            project_status="active",
            project_updated="2026-07-03",
        ),
        SourceRecord(
            "projects/paused/CURRENT.md",
            owner,
            "project:paused",
            project_status="paused",
            project_updated="2026-09-30",
        ),
        SourceRecord("research/current.md", owner, "research"),
        SourceRecord("research/large-body.md", owner, "research"),
        SourceRecord("research/no-status.md", owner, "research"),
    ]


def test_source_manifest_is_exact_deterministic_and_path_safe(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    build_fixture(vault)
    cfg = Config(vault_path=str(vault), source_policy_owner="owner-a")
    opened: list[str] = []
    chars_read: dict[str, int] = {}
    original_open = Path.open

    class TrackedFile:
        def __init__(self, relative: str, handle):
            self.relative = relative
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def readline(self, *args, **kwargs):
            line = self.handle.readline(*args, **kwargs)
            chars_read[self.relative] = chars_read.get(self.relative, 0) + len(line)
            return line

    def tracked_open(path, *args, **kwargs):
        relative = path.relative_to(vault).as_posix()
        opened.append(relative)
        return TrackedFile(relative, original_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", tracked_open)

    first = build_source_manifest(config=cfg, today=TODAY)
    second = build_source_manifest(config=cfg, today=TODAY)

    assert first == normal_expected()
    assert second == first
    assert [record.relative_path for record in first] == sorted(
        record.relative_path for record in first
    )
    assert len({record.relative_path for record in first}) == len(first)
    assert all(record.owner and record.namespace for record in first)
    assert all(".." not in Path(record.relative_path).parts for record in first)
    assert set(opened) <= {
        "research/current.md",
        "research/large-body.md",
        "research/no-status.md",
        "research/old-active.md",
        "research/archived.md",
        "research/superseded.md",
        "projects/active/specs/CURRENT.md",
        "projects/paused/CURRENT.md",
        "projects/stale/CURRENT.md",
        "projects/future/CURRENT.md",
        "projects/completed/CURRENT.md",
        "projects/abandoned/CURRENT.md",
        "projects/archived/CURRENT.md",
        "projects/superseded/CURRENT.md",
        "projects/malformed-date/CURRENT.md",
        "projects/missing-date/CURRENT.md",
        "projects/missing-status/CURRENT.md",
        "projects/unknown-status/CURRENT.md",
    }
    assert "sensitive/CURRENT.md" not in opened
    assert "business/logs/incident.md" not in opened
    assert "business/service-log.md" not in opened
    assert "business/error.log.md" not in opened
    assert chars_read["research/large-body.md"] < 18000


def test_old_research_is_excluded_from_manifest_without_historical_mode(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    build_fixture(vault)
    cfg = Config(vault_path=str(vault), source_policy_owner="owner-a")

    manifest = build_source_manifest(config=cfg, today=TODAY)

    assert manifest == normal_expected()
    assert "historical" not in signature(build_source_manifest).parameters
    assert "old-active.md" not in {record.relative_path for record in manifest}
    assert (vault / "research" / "old-active.md").exists()
    assert (vault / "research" / "archived.md").exists()
    assert (vault / "research" / "superseded.md").exists()
    record_fields = set(SourceRecord.__dataclass_fields__)
    assert record_fields == {
        "relative_path",
        "owner",
        "namespace",
        "project_status",
        "project_updated",
    }
    assert all(set(asdict(record)) == record_fields for record in manifest)


def test_source_policy_config_controls_owner_roots_and_stale_threshold(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    build_fixture(vault)
    cfg = Config(
        vault_path=str(vault),
        source_policy_owner="custom-owner",
        source_policy_project_stale_days=1,
        source_policy_include_roots=["AGENTS.md", "projects"],
    )

    manifest = build_source_manifest(config=cfg, today=TODAY)

    assert [asdict(record) for record in manifest] == [
        asdict(SourceRecord("AGENTS.md", "custom-owner", "workspace")),
        asdict(
            SourceRecord(
                "projects/paused/CURRENT.md",
                "custom-owner",
                "project:paused",
                project_status="paused",
                project_updated="2026-09-30",
            )
        ),
    ]


def test_source_policy_config_env_fields_reset_singleton(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("MEMORY_V3_VAULT", str(vault))
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "env-owner")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_PROJECT_STALE_DAYS", "7")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_RESEARCH_RECENCY_DAYS", "14")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_INCLUDE_ROOTS", "AGENTS.md,projects")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_DENY_ROOTS", "inbox,logs")
    reset_config()

    cfg = get_config()

    assert cfg.vault_path == str(vault)
    assert cfg.source_policy_owner == "env-owner"
    assert cfg.source_policy_project_stale_days == 7
    assert cfg.source_policy_research_recency_days == 14
    assert cfg.source_policy_include_roots == ["AGENTS.md", "projects"]
    assert cfg.source_policy_deny_roots == ["inbox", "logs"]

    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "changed-owner")
    assert get_config().source_policy_owner == "env-owner"
    reset_config()
    assert get_config().source_policy_owner == "changed-owner"
    reset_config()
