from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from memory_v3.config import Config, get_config

PROJECT_STATUSES = {"active", "paused", "completed", "abandoned", "archived", "superseded"}
TERMINAL_PROJECT_STATUSES = {"completed", "abandoned", "archived", "superseded"}
INACTIVE_RESEARCH_STATUSES = {"archived", "superseded"}
TEMP_SUFFIXES = {".log", ".tmp", ".temp", ".bak", ".swp", ".old", ".orig"}
LOG_DIR_NAMES = {"log", "logs"}
HEADER_MAX_LINES = 40
HEADER_MAX_BYTES = 8192


@dataclass(frozen=True)
class SourceRecord:
    relative_path: str
    owner: str
    namespace: str
    project_status: str | None = None
    project_updated: str | None = None


def build_source_manifest(
    vault_path: str | Path | None = None,
    *,
    config: Config | None = None,
    today: date | None = None,
) -> list[SourceRecord]:
    cfg = config or get_config()
    root = _vault_root(vault_path or cfg.vault_path)
    current_date = today or date.today()
    records: list[SourceRecord] = []

    for path in _candidate_files(root, cfg):
        relative = _safe_relative_path(root, path)
        if relative is None:
            continue

        record = _classify_path(
            root,
            relative,
            cfg,
            today=current_date,
        )
        if record is not None:
            records.append(record)

    return sorted(records, key=lambda record: record.relative_path)


def classify_source_path(
    path: str | Path,
    vault_path: str | Path | None = None,
    *,
    config: Config | None = None,
    today: date | None = None,
) -> SourceRecord | None:
    cfg = config or get_config()
    root = _vault_root(vault_path or cfg.vault_path)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if _has_symlink_component(root, candidate):
        return None
    relative = _safe_relative_path(root, candidate)
    if relative is None or _excluded_file(root, candidate, candidate.name):
        return None
    return _classify_path(
        root,
        relative,
        cfg,
        today=today or date.today(),
    )


def make_source_id(owner: str, relative_path: str) -> str:
    canonical_path = _canonical_relative_path(relative_path)
    digest = hashlib.sha256(f"{owner}\0{canonical_path}".encode()).hexdigest()
    return f"src:{digest[:32]}"


def make_chunk_id(source_id: str, content_hash: str, chunk_index: int = 0) -> str:
    digest = hashlib.sha256(f"{source_id}\0{content_hash}\0{chunk_index}".encode()).hexdigest()
    return f"chk:{digest[:32]}"


def _canonical_relative_path(relative_path: str) -> str:
    return Path(relative_path).as_posix().lstrip("/")


def _vault_root(vault_path: str | Path) -> Path:
    if not vault_path:
        raise ValueError("vault_path is required")
    root = Path(vault_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"vault_path is not a directory: {root}")
    return root


def _candidate_files(root: Path, cfg: Config):
    deny_roots = {item.casefold() for item in cfg.source_policy_deny_roots}
    include_roots = {item.rstrip("/") for item in cfg.source_policy_include_roots}
    include_dirs = {item for item in include_roots if "." not in Path(item).name}
    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(current_root)
        relative_dir = _safe_relative_path(root, current)
        if relative_dir is None:
            dirnames[:] = []
            continue

        if relative_dir == ".":
            dirnames[:] = [dirname for dirname in dirnames if dirname in include_dirs]

        dirnames[:] = [
            dirname
            for dirname in sorted(dirnames)
            if not _excluded_dir(current / dirname, dirname, deny_roots)
        ]

        for filename in sorted(filenames):
            path = current / filename
            if _excluded_file(root, path, filename):
                continue
            yield path


def _excluded_dir(path: Path, name: str, deny_roots: set[str]) -> bool:
    folded = name.casefold()
    return (
        path.is_symlink()
        or name.startswith(".")
        or folded in deny_roots
        or folded in LOG_DIR_NAMES
        or folded == "archive"
        or folded == "lifeos"
        or folded == "shared"
    )


def _excluded_file(root: Path, path: Path, name: str) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        return True
    relative = _safe_relative_path(root, path)
    if relative is None:
        return True
    parts = Path(relative).parts
    folded_parts = [part.casefold() for part in parts]
    suffix = path.suffix.casefold()
    return (
        name.startswith(".")
        or any(part.startswith(".") for part in parts)
        or "archive" in folded_parts
        or "shared" in folded_parts
        or "lifeos" in folded_parts
        or "scratchpad" in path.name.casefold()
        or _is_log_name(path.name)
        or suffix in TEMP_SUFFIXES
        or name.endswith("~")
    )


def _has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError:
        return False

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _is_log_name(name: str) -> bool:
    stem = Path(name).stem.casefold()
    folded = name.casefold()
    return (
        folded in {"log.md", "logs.md"}
        or stem.endswith("-log")
        or stem.endswith("_log")
        or stem.endswith(".log")
    )


def _safe_relative_path(root: Path, path: Path) -> str | None:
    try:
        resolved = path.resolve(strict=False)
        relative = resolved.relative_to(root)
    except ValueError:
        return None
    return relative.as_posix()


def _classify_path(
    root: Path,
    relative: str,
    cfg: Config,
    *,
    today: date,
) -> SourceRecord | None:
    parts = Path(relative).parts
    if not parts:
        return None

    include_roots = {item.rstrip("/") for item in cfg.source_policy_include_roots}
    first = parts[0]

    if len(parts) == 1:
        if first in include_roots:
            return _record(relative, cfg.source_policy_owner, "workspace")
        return None

    if first not in include_roots:
        return None

    if first == "business":
        return _record(relative, cfg.source_policy_owner, "business")
    if first == "decisions":
        return _record(relative, cfg.source_policy_owner, "decisions")
    if first == "infrastructure":
        return _record(relative, cfg.source_policy_owner, "infrastructure")
    if first == "docs":
        return _record(relative, cfg.source_policy_owner, "docs")
    if first == "research":
        return _classify_research(
            root / relative,
            relative,
            cfg,
            today=today,
        )
    if first == "projects":
        return _classify_project(root / relative, relative, cfg, today=today)

    return None


def _classify_research(
    path: Path,
    relative: str,
    cfg: Config,
    *,
    today: date,
) -> SourceRecord | None:
    header = _read_header(path)
    status = _extract_status(header)
    if status in INACTIVE_RESEARCH_STATUSES:
        return None
    updated = _extract_last_updated(header) or _extract_frontmatter_updated(header)
    if updated is None:
        updated = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date()
    age_days = (today - updated).days
    if age_days > cfg.source_policy_research_recency_days:
        return None
    return _record(relative, cfg.source_policy_owner, "research")


def _classify_project(
    path: Path,
    relative: str,
    cfg: Config,
    *,
    today: date,
) -> SourceRecord | None:
    parts = Path(relative).parts
    if len(parts) < 3 or parts[-1] != "CURRENT.md":
        return None

    slug = parts[1]
    if not slug or slug in {".", ".."} or "/" in slug:
        return None

    text = _read_header(path)
    if not text.strip():
        return None

    status = _extract_status(text)
    if status not in PROJECT_STATUSES or status in TERMINAL_PROJECT_STATUSES:
        return None

    updated = _extract_last_updated(text)
    if updated is None:
        return None
    age_days = (today - updated).days
    if age_days < 0 or age_days > cfg.source_policy_project_stale_days:
        return None

    return _record(
        relative,
        cfg.source_policy_owner,
        f"project:{slug}",
        project_status=status,
        project_updated=updated.isoformat(),
    )


def _read_status(path: Path) -> str | None:
    return _extract_status(_read_header(path))


def _read_header(path: Path) -> str:
    lines: list[str] = []
    bytes_read = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(HEADER_MAX_LINES):
            line = handle.readline(HEADER_MAX_BYTES - bytes_read + 1)
            if not line:
                break
            bytes_read += len(line.encode("utf-8"))
            if bytes_read > HEADER_MAX_BYTES:
                break
            lines.append(line)
    return "".join(lines)


def _extract_status(text: str) -> str | None:
    for line in text.splitlines()[:40]:
        match = re.match(r"^\s*(?:[-*]\s*)?(?:status|Status)\s*:\s*(.+?)\s*$", line)
        if match:
            value = match.group(1).strip().lower()
            return re.split(r"[.;,]|\s+-\s+", value, maxsplit=1)[0].strip()
    return None


def _extract_last_updated(text: str) -> date | None:
    for line in text.splitlines()[:40]:
        match = re.match(r"^\s*Last updated\s*:\s*(\d{4}-\d{2}-\d{2})\s*$", line, re.I)
        if not match:
            continue
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _extract_frontmatter_updated(text: str) -> date | None:
    lines = text.splitlines()[:40]
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            return None
        match = re.match(r"^\s*updated\s*:\s*(\d{4}-\d{2}-\d{2})\s*$", line, re.I)
        if not match:
            continue
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _record(
    relative_path: str,
    owner: str,
    namespace: str,
    *,
    project_status: str | None = None,
    project_updated: str | None = None,
) -> SourceRecord:
    return SourceRecord(
        relative_path=relative_path,
        owner=owner,
        namespace=namespace,
        project_status=project_status,
        project_updated=project_updated,
    )
