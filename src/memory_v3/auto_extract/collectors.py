from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class CollectedPayload:
    trigger_type: str
    fingerprint: str
    text: str
    source: str
    source_record: object | None = None
    source_hash: str | None = None
    source_updated_at: str | None = None
    source_records: tuple[object, ...] = ()


def _run_git(
    git_args: list[str], *, git_dir: str | None, work_tree: str | None
) -> str:
    cmd = ["git"]
    if git_dir:
        cmd += ["--git-dir", git_dir]
    if work_tree:
        cmd += ["--work-tree", work_tree]
    cmd += git_args
    env = os.environ.copy()
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def collect_git_payload(
    *,
    fingerprint: str,
    git_dir: str | None,
    work_tree: str | None,
    max_chars: int = 20000,
) -> CollectedPayload:
    patch_plus_meta = _run_git(
        [
            "show",
            fingerprint,
            "--no-color",
            "--patch",
            "--stat",
            "--format=%B",
        ],
        git_dir=git_dir,
        work_tree=work_tree,
    ).strip()

    if not patch_plus_meta:
        return CollectedPayload(
            trigger_type="git",
            fingerprint=fingerprint,
            text="",
            source="auto_extract:git",
        )

    text = patch_plus_meta
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[TRUNCATED]"

    return CollectedPayload(
        trigger_type="git",
        fingerprint=fingerprint,
        text=text,
        source="auto_extract:git",
    )


def compute_git_wip_fingerprint(
    *,
    git_dir: str | None,
    work_tree: str | None,
    max_chars: int = 20000,
) -> str:
    diff = _run_git(
        ["diff", "--no-color", "HEAD"],
        git_dir=git_dir,
        work_tree=work_tree,
    )
    if not diff:
        diff = ""
    if len(diff) > max_chars:
        diff = diff[:max_chars] + "\n[TRUNCATED]"
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def collect_git_wip_payload(
    *,
    fingerprint: str,
    git_dir: str | None,
    work_tree: str | None,
    max_chars: int = 20000,
) -> CollectedPayload:
    name_list = _run_git(
        ["diff", "--name-only", "HEAD"],
        git_dir=git_dir,
        work_tree=work_tree,
    ).strip()

    diff = _run_git(
        ["diff", "--no-color", "HEAD"],
        git_dir=git_dir,
        work_tree=work_tree,
    )
    diff = (diff or "").strip()
    if len(diff) > max_chars:
        diff = diff[:max_chars] + "\n[TRUNCATED]"

    parts: list[str] = []
    if name_list:
        parts.append("[CHANGED_FILES]\n" + name_list)
    if diff:
        parts.append("[GIT_DIFF]\n" + diff)

    text = "\n\n".join(parts).strip()

    return CollectedPayload(
        trigger_type="git-wip",
        fingerprint=fingerprint,
        text=text,
        source="auto_extract:git-wip",
    )


def default_daily_fingerprint() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def iter_vault_markdown_files(
    vault_dir: str,
    since_dt,
    *,
    max_files: int = 50,
    config=None,
):
    from pathlib import Path

    from ..source_policy import build_source_manifest

    root = Path(vault_dir)
    if not root.exists() or not root.is_dir():
        return

    items = []
    for record in build_source_manifest(root, config=config):
        p = root / record.relative_path
        try:
            st = p.stat()
        except OSError:
            continue
        mtime = st.st_mtime
        mtime_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        if since_dt is None or mtime_dt > since_dt:
            items.append((mtime, p, record))

    items.sort(key=lambda x: x[0], reverse=True)
    for _, p, record in items[:max_files]:
        yield p, record


def compute_vault_fingerprint(
    *,
    vault_dir: str,
    since_dt,
    max_files: int = 50,
    max_chars_per_file: int = 40000,
    config=None,
) -> str:
    """Deterministic fingerprint for vault changes since since_dt."""
    import hashlib

    parts: list[str] = []
    for p, record in iter_vault_markdown_files(
        vault_dir, since_dt, max_files=max_files, config=config
    ):
        st = p.stat()
        mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        size = st.st_size
        parts.append(f"{record.relative_path}|{mtime_iso}|{size}")

    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _vault_payload_for_record(
    *,
    fingerprint: str,
    path,
    record,
    max_chars_per_file: int,
) -> CollectedPayload | None:
    try:
        full_text = path.read_text(encoding="utf-8")
        st = path.stat()
    except Exception:
        return None

    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    text = full_text
    if len(text) > max_chars_per_file:
        text = text[:max_chars_per_file] + "\n[TRUNCATED]"
    mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    payload_text = "\n".join(
        [
            f"[FILE] {record.relative_path}",
            f"[OWNER] {record.owner}",
            f"[NAMESPACE] {record.namespace}",
            f"[MTIME] {mtime_iso}",
            "[CONTENT]",
            text,
        ]
    )
    return CollectedPayload(
        trigger_type="vault",
        fingerprint=fingerprint,
        text=payload_text,
        source=record.relative_path,
        source_record=record,
        source_hash=content_hash,
        source_updated_at=mtime_iso,
    )


def collect_vault_payloads(
    *,
    fingerprint: str,
    vault_dir: str,
    since_dt,
    max_files: int = 50,
    max_chars_per_file: int = 40000,
    config=None,
) -> list[CollectedPayload]:
    payloads: list[CollectedPayload] = []
    for path, record in iter_vault_markdown_files(
        vault_dir, since_dt, max_files=max_files, config=config
    ):
        payload = _vault_payload_for_record(
            fingerprint=fingerprint,
            path=path,
            record=record,
            max_chars_per_file=max_chars_per_file,
        )
        if payload is not None:
            payloads.append(payload)
    return payloads


def collect_vault_payload(
    *,
    fingerprint: str,
    vault_dir: str,
    since_dt,
    max_files: int = 50,
    max_chars_per_file: int = 40000,
    config=None,
) -> CollectedPayload:
    payloads = collect_vault_payloads(
        fingerprint=fingerprint,
        vault_dir=vault_dir,
        since_dt=since_dt,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
        config=config,
    )
    text = "\n\n".join(payload.text for payload in payloads).strip()
    return CollectedPayload(
        trigger_type="vault",
        fingerprint=fingerprint,
        text=text,
        source="auto_extract:vault",
        source_records=tuple(
            payload.source_record for payload in payloads if payload.source_record is not None
        ),
    )
