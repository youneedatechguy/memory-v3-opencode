from __future__ import annotations

import hashlib
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import db

DEFAULT_TARGET_CHARS = 3500
DEFAULT_MAX_CHARS = 5000
DEFAULT_OVERLAP_LINES = 3


@dataclass(frozen=True)
class MarkdownChunk:
    source_id: str
    source_version_key: str
    chunk_id: str
    chunk_index: int
    chunk_count: int
    owner: str
    namespace: str
    relative_path: str
    source_hash: str
    line_start: int
    line_end: int
    heading_path: tuple[str, ...]
    body: str
    embedding_text: str


@dataclass(frozen=True)
class PreparedChunk:
    chunk: MarkdownChunk
    embedding_blob: bytes


@dataclass(frozen=True)
class SourceIndexRecord:
    relative_path: str
    owner: str
    namespace: str
    project_status: str | None = None
    project_updated: str | None = None


def reindex_sources(
    conn,
    vault_path: str | Path,
    records,
    embedder_factory,
    *,
    embed_model: str,
    embed_provider: str,
    embed_dim: int,
    force: bool,
) -> dict:
    from .scoring.hierarchy import classify_governance_layer

    vault = Path(vault_path).expanduser().resolve(strict=True)
    result = {
        "indexed": 0,
        "skipped": 0,
        "errors": 0,
        "already_current": 0,
        "force": force,
        "source_errors": [],
    }
    embedder = None

    for record in records:
        rel_path = record.relative_path
        try:
            before_hash = _active_file_hash(conn, rel_path)
            md_file = _safe_source_path(vault, rel_path)
            content = md_file.read_text(encoding="utf-8")
            source_mtime = datetime.fromtimestamp(md_file.stat().st_mtime, timezone.utc).isoformat()
            chunks = build_source_chunks(record.owner, record.namespace, rel_path, content)
        except Exception as exc:
            result["errors"] += 1
            result["source_errors"].append(
                _source_error(rel_path, None, "read", exc, False, True, True)
            )
            continue

        if not chunks:
            result["skipped"] += 1
            continue

        first = chunks[0]
        if _source_is_current(conn, record.owner, chunks):
            result["already_current"] += 1
            result["skipped"] += 1
            if force:
                result["status"] = "already_current"
            continue

        before_source_rows = _source_row_signature(conn, first.source_id, record.owner)
        try:
            if embedder is None:
                embedder = embedder_factory()
            prepared = _embed_all_chunks(embedder, chunks, embed_dim)
            content_type = _detect_index_content_type(content)
            tags = _extract_index_tags(content)
            governance_layer = classify_governance_layer(content_type, tags=set(tags))
            db.replace_source_chunks(
                conn,
                owner=record.owner,
                namespace=record.namespace,
                source_id=first.source_id,
                source_path=first.relative_path,
                source_hash=first.source_hash,
                source_mtime=source_mtime,
                chunks=[
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "chunk_index": item.chunk.chunk_index,
                        "chunk_count": item.chunk.chunk_count,
                        "content": item.chunk.embedding_text,
                        "source_line": item.chunk.line_start,
                        "embedding": item.embedding_blob,
                    }
                    for item in prepared
                ],
                content_type=content_type,
                tags=tags,
                governance_layer=governance_layer,
                project_status=record.project_status,
                project_updated_at=record.project_updated,
                embed_model=embed_model,
                embed_provider=embed_provider,
                embed_dim=embed_dim,
            )
            result["indexed"] += 1
        except Exception as exc:
            result["errors"] += 1
            result["source_errors"].append(
                _source_error(
                    rel_path,
                    first.source_id,
                    "replace_source",
                    exc,
                    True,
                    _source_rows_match(conn, first.source_id, record.owner, before_source_rows),
                    _active_file_hash(conn, rel_path) == before_hash,
                )
            )

    if not result["source_errors"]:
        result.pop("source_errors")
    if not result["already_current"]:
        result.pop("already_current")
    return result


def _safe_source_path(vault: Path, relative_path: str) -> Path:
    canonical = canonical_relative_path(relative_path)
    path = (vault / canonical).resolve(strict=True)
    path.relative_to(vault)
    if not path.is_file():
        raise ValueError("source path is not a file")
    return path


def _source_is_current(
    conn,
    owner: str,
    chunks: tuple[MarkdownChunk, ...],
) -> bool:
    first = chunks[0]
    if _active_file_hash(conn, first.relative_path) != first.source_hash:
        return False
    rows = conn.execute(
        """
        SELECT source_hash, source_id, chunk_id, chunk_index, chunk_count
        FROM memories
        WHERE owner = ? AND source_id = ? AND record_kind = 'derived' AND archived = 0
        ORDER BY chunk_index, id
        """,
        (owner, first.source_id),
    ).fetchall()
    if len(rows) != len(chunks):
        return False
    for row, chunk in zip(rows, chunks, strict=True):
        if row["source_hash"] != first.source_hash:
            return False
        if row["source_id"] != first.source_id:
            return False
        if row["chunk_id"] != chunk.chunk_id:
            return False
        if row["chunk_index"] != chunk.chunk_index:
            return False
        if row["chunk_count"] != len(chunks):
            return False
    return True


def _source_row_signature(conn, source_id: str, owner: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT COUNT(*) AS count, COALESCE(MAX(id), 0) AS max_id "
        "FROM memories WHERE source_id = ? AND owner = ? AND record_kind = 'derived'",
        (source_id, owner),
    ).fetchone()
    if row is None:
        return (0, 0)
    return (row["count"], row["max_id"])


def _source_rows_match(conn, source_id: str, owner: str, expected: tuple[int, int]) -> bool:
    return _source_row_signature(conn, source_id, owner) == expected


def _embed_all_chunks(
    embedder, chunks: tuple[MarkdownChunk, ...], embed_dim: int
) -> tuple[PreparedChunk, ...]:
    prepared = tuple(
        PreparedChunk(
            chunk=chunk,
            embedding_blob=_serialize_checked_embedding(
                embedder.embed(chunk.embedding_text), embed_dim
            ),
        )
        for chunk in chunks
    )
    if len(prepared) != len(chunks):
        raise ValueError("embedding batch cardinality mismatch")
    return prepared


def _serialize_checked_embedding(embedding: list[float], expected_dim: int) -> bytes:
    array = np.asarray(embedding, dtype=np.float32)
    if array.shape != (expected_dim,) or not np.isfinite(array).all():
        raise ValueError("invalid embedding")
    return array.tobytes()


def _active_file_hash(conn, relative_path: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM file_hashes WHERE file_path = ?",
        (canonical_relative_path(relative_path),),
    ).fetchone()
    return None if row is None else row["content_hash"]


def _source_error(
    source_path: str,
    source_id: str | None,
    stage: str,
    exc: Exception,
    retryable: bool,
    source_rows_unchanged: bool,
    file_hash_unchanged: bool,
) -> dict:
    category = "provider_failure"
    if stage == "read":
        category = "read_failure"
    if stage == "replace_source":
        category = "replacement_failure"
    return {
        "source_path": source_path,
        "source_id": source_id,
        "stage": stage,
        "retryable": retryable,
        "source_rows_unchanged": source_rows_unchanged,
        "file_hash_unchanged": file_hash_unchanged,
        "error_category": category,
        "exception_class": exc.__class__.__name__,
    }


def sanitized_error(exc: Exception, category: str) -> dict:
    return {
        "error_category": category,
        "exception_class": exc.__class__.__name__,
    }


def _detect_index_content_type(text: str) -> str:
    lowered = text.lower()
    if any(
        keyword in lowered
        for keyword in ("decided", "decision", "chose", "will use", "going with", "committed to")
    ):
        return "decision"
    if any(
        keyword in lowered for keyword in ("correction", "corrected", "actually no", "not true")
    ):
        return "correction"
    if any(keyword in lowered for keyword in ("my name is", "i am a", "i'm a", "my role is")):
        return "identity"
    if any(
        keyword in lowered
        for keyword in ("person:", "family:", "colleague:", "friend:", "partner:")
    ):
        return "person"
    return "fact"


def _extract_index_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = []
    for keyword in (
        "python",
        "javascript",
        "typescript",
        "rust",
        "docker",
        "git",
        "api",
        "database",
        "sqlite",
        "memory",
        "preference",
        "identity",
        "decision",
        "correction",
        "project",
    ):
        if keyword in lowered:
            tags.append(keyword)
    content_type = _detect_index_content_type(text)
    if content_type not in tags:
        tags.append(content_type)
    return tags


def rebuild_graph_debt(
    conn,
    *,
    graph_dir: str | Path | None = None,
    extraction_fn=None,
    publish_hook=None,
) -> dict:
    return {
        "status": "disabled",
        "deprecated": True,
        "code": "graph_lifecycle_disabled",
        "error": "Legacy graph rebuild is disabled; active indexing is graph-free.",
        "data": [],
    }


def build_source_chunks(
    owner: str,
    namespace: str,
    relative_path: str,
    content: str,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> tuple[MarkdownChunk, ...]:
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if target_chars > max_chars:
        raise ValueError("target_chars must not exceed max_chars")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must not be negative")

    canonical_path = canonical_relative_path(relative_path)
    if content == "":
        return ()
    source_hash = hash_text(content)
    source_id = make_index_source_id(owner, namespace, canonical_path)
    source_version_key = make_source_version_key(source_id, source_hash)
    lines = content.splitlines(keepends=True)
    headings = _heading_paths_by_line(lines)
    spans = _chunk_spans(
        lines,
        canonical_path,
        headings,
        target_chars,
        max_chars,
        overlap_lines,
    )

    chunks: list[MarkdownChunk] = []
    for index, span in enumerate(spans):
        start, end = span
        body = "".join(lines[start:end])
        line_start = start + 1
        line_end = end
        heading_path = headings[start] if headings else ()
        chunk_id = make_index_chunk_id(
            source_version_key,
            index,
            line_start,
            line_end,
            hash_text(body),
        )
        chunks.append(
            MarkdownChunk(
                source_id=source_id,
                source_version_key=source_version_key,
                chunk_id=chunk_id,
                chunk_index=index,
                chunk_count=len(spans),
                owner=owner,
                namespace=namespace,
                relative_path=canonical_path,
                source_hash=source_hash,
                line_start=line_start,
                line_end=line_end,
                heading_path=heading_path,
                body=body,
                embedding_text=format_chunk_text(
                    canonical_path,
                    line_start,
                    line_end,
                    heading_path,
                    body,
                ),
            )
        )
    return tuple(chunks)


def canonical_relative_path(relative_path: str) -> str:
    if relative_path == "" or relative_path.strip() == "":
        raise ValueError("relative_path must not be empty")
    if relative_path.startswith(("/", "\\")):
        raise ValueError("relative_path must not be absolute")
    if re.match(r"^[A-Za-z]:", relative_path):
        raise ValueError("relative_path must not contain a Windows drive")
    if relative_path.startswith(("//", "\\\\")):
        raise ValueError("relative_path must not be a UNC path")

    normalized = relative_path.replace("\\", "/")
    parts = normalized.split("/")
    if ".." in parts:
        raise ValueError("relative_path must not contain traversal")
    canonical = posixpath.normpath(normalized).lstrip("/")
    if canonical in {"", "."}:
        raise ValueError("relative_path must identify a file")
    if ".." in canonical.split("/"):
        raise ValueError("relative_path must not contain traversal")
    return canonical


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def make_index_source_id(owner: str, namespace: str, relative_path: str) -> str:
    canonical_path = canonical_relative_path(relative_path)
    digest = hashlib.sha256(f"{owner}\0{namespace}\0{canonical_path}".encode()).hexdigest()
    return f"src:{digest}"


def make_source_version_key(source_id: str, source_hash: str) -> str:
    digest = hashlib.sha256(f"{source_id}\0{source_hash}".encode()).hexdigest()
    return f"ver:{digest}"


def make_index_chunk_id(
    source_version_key: str,
    chunk_index: int,
    line_start: int,
    line_end: int,
    body_hash: str,
) -> str:
    digest = hashlib.sha256(
        f"{source_version_key}\0{chunk_index}\0{line_start}\0{line_end}\0{body_hash}".encode()
    ).hexdigest()
    return f"chk:{digest}"


def format_chunk_text(
    relative_path: str,
    line_start: int,
    line_end: int,
    heading_path: tuple[str, ...],
    body: str,
) -> str:
    headings = " > ".join(heading_path) if heading_path else "(none)"
    return (
        f"Source: {relative_path}\nLines: {line_start}-{line_end}\nHeadings: {headings}\n\n{body}"
    )


def _chunk_spans(
    lines: list[str],
    relative_path: str,
    headings: list[tuple[str, ...]],
    target_chars: int,
    max_chars: int,
    overlap_lines: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    line_count = len(lines)
    while start < line_count:
        end = start
        body_size = 0
        while end < line_count:
            candidate_body = "".join(lines[start : end + 1])
            candidate_text = format_chunk_text(
                relative_path,
                start + 1,
                end + 1,
                headings[start] if headings else (),
                candidate_body,
            )
            if end == start and len(candidate_text) > max_chars:
                end += 1
                break
            if end > start and len(candidate_text) > max_chars:
                break
            body_size = len(candidate_body)
            end += 1
            if body_size >= target_chars:
                break

        if end == start:
            end += 1
        spans.append((start, end))
        if end == line_count:
            break

        start = _next_start(
            lines,
            relative_path,
            headings,
            start,
            end,
            max_chars,
            overlap_lines,
        )
    return spans


def _next_start(
    lines: list[str],
    relative_path: str,
    headings: list[tuple[str, ...]],
    start: int,
    end: int,
    max_chars: int,
    overlap_lines: int,
) -> int:
    next_start = max(start + 1, end - overlap_lines)
    while next_start < end:
        overlap_body = "".join(lines[next_start:end])
        overlap_text = format_chunk_text(
            relative_path,
            next_start + 1,
            end,
            headings[next_start] if headings else (),
            overlap_body,
        )
        if len(overlap_text) <= max_chars:
            return next_start
        next_start += 1
    return end


def _heading_paths_by_line(lines: list[str]) -> list[tuple[str, ...]]:
    stack: list[str] = []
    paths: list[tuple[str, ...]] = []
    for line in lines:
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*(?:\r?\n)?$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
        paths.append(tuple(stack))
    return paths
