import subprocess
from pathlib import Path


class StubLLM:
    def chat(self, prompt, max_tokens=2048):
        return "- [2026-05-03] Auto-extract WIP stored fact for testing"


def _git(command: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout.strip()


def test_auto_extract_git_wip_dedup(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["git", "init"], cwd=repo)
    (repo / "file.txt").write_text("hello\n", encoding="utf-8")
    _git(["git", "add", "file.txt"], cwd=repo)
    _git(["git", "commit", "-m", "base"], cwd=repo)

    # WIP change (no new commit)
    (repo / "file.txt").write_text("hello\nworld\n", encoding="utf-8")

    git_dir = str(repo / ".git")
    work_tree = str(repo)
    db_path = tmp_path / "memory.db"
    state_path = tmp_path / "auto_extract_state.json"

    monkeypatch.setenv("MEMORY_V3_DB", str(db_path))
    monkeypatch.setenv("MEMORY_V3_ENABLE_SENSORY", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_SURPRISE", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_FAN_EFFECT", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_THREE_STAGE", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_HALUMEM", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_CONSOLIDATION", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_ZETTELKASTEN", "false")
    monkeypatch.setenv("MEMORY_V3_ENABLE_MULTIGRAPH", "false")
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "true")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    import memory_v3.lifecycle.extraction as extraction

    monkeypatch.setattr(extraction, "get_llm", lambda: StubLLM())

    from memory_v3.db import init_db

    conn = init_db(str(db_path))

    from memory_v3.auto_extract.runner import AutoExtractRunner

    runner = AutoExtractRunner(conn=conn, config=cfg, state_path=state_path)

    res1 = runner.run(
        trigger_type="git-wip",
        fingerprint="auto",
        git_dir=git_dir,
        work_tree=work_tree,
        dry_run=False,
        force=False,
    )

    assert res1.skipped is False
    assert res1.extracted.get("errors", 0) == 0
    row_count_1 = conn.execute("SELECT COUNT(*) FROM memories WHERE archived = 0").fetchone()[0]
    assert row_count_1 >= 1

    res2 = runner.run(
        trigger_type="git-wip",
        fingerprint="auto",
        git_dir=git_dir,
        work_tree=work_tree,
        dry_run=False,
        force=False,
    )

    assert res2.skipped is True
    assert "Already extracted" in res2.reason
    row_count_2 = conn.execute("SELECT COUNT(*) FROM memories WHERE archived = 0").fetchone()[0]
    assert row_count_2 == row_count_1

    conn.close()


def test_auto_extract_git_wip_disabled_skips_fingerprint_collectors_state_db_and_processing(
    tmp_path: Path, monkeypatch
):
    db_path = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_V3_DB", str(db_path))
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    import memory_v3.auto_extract.runner as runner_mod

    calls = []

    def fail_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("disabled git-wip auto-extract should fail closed")

    monkeypatch.setattr(runner_mod, "compute_git_wip_fingerprint", fail_call)
    monkeypatch.setattr(runner_mod, "collect_git_wip_payload", fail_call)
    monkeypatch.setattr(runner_mod, "init_db", fail_call)
    monkeypatch.setattr(runner_mod, "process_conversation", fail_call)
    monkeypatch.setattr(runner_mod.AutoExtractState, "get", fail_call)
    monkeypatch.setattr(runner_mod.AutoExtractState, "mark_done", fail_call)

    runner = runner_mod.AutoExtractRunner(config=cfg, state_path=tmp_path / "state.json")
    result = runner.run(
        trigger_type="git-wip",
        fingerprint="auto",
        git_dir=str(tmp_path / ".git"),
        work_tree=str(tmp_path),
    )

    assert result.skipped is True
    assert result.reason == "Direct writes are disabled"
    assert result.collected_chars == 0
    assert result.extracted == {}
    assert calls == []
    assert not db_path.exists()
    assert not (tmp_path / "state.json").exists()
    reset_config()


def test_auto_extract_daily_disabled_skips_fingerprint_collectors_state_db_and_processing(
    tmp_path: Path, monkeypatch
):
    db_path = tmp_path / "memory.db"
    monkeypatch.setenv("MEMORY_V3_DB", str(db_path))
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "false")

    from memory_v3.config import get_config, reset_config

    reset_config()
    cfg = get_config()

    import memory_v3.auto_extract.runner as runner_mod

    calls = []

    def fail_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("disabled daily auto-extract should fail closed")

    monkeypatch.setattr(runner_mod, "default_daily_fingerprint", fail_call)
    monkeypatch.setattr(runner_mod, "collect_git_wip_payload", fail_call)
    monkeypatch.setattr(runner_mod, "init_db", fail_call)
    monkeypatch.setattr(runner_mod, "process_conversation", fail_call)
    monkeypatch.setattr(runner_mod.AutoExtractState, "get", fail_call)
    monkeypatch.setattr(runner_mod.AutoExtractState, "mark_done", fail_call)

    runner = runner_mod.AutoExtractRunner(config=cfg, state_path=tmp_path / "state.json")
    result = runner.run(
        trigger_type="daily",
        fingerprint="auto",
        git_dir=str(tmp_path / ".git"),
        work_tree=str(tmp_path),
    )

    assert result.skipped is True
    assert result.reason == "Direct writes are disabled"
    assert result.collected_chars == 0
    assert result.extracted == {}
    assert calls == []
    assert not db_path.exists()
    assert not (tmp_path / "state.json").exists()
    reset_config()
