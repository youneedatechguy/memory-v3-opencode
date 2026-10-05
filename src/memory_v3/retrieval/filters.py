from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

ELIGIBLE_PROJECT_STATUSES = ("active", "paused")


@dataclass(frozen=True)
class RetrievalFilter:
    owner: str | None = None
    namespaces: tuple[str, ...] = field(default_factory=tuple)
    project_slug: str | None = None
    path_prefix: str | None = None
    source_id: str | None = None
    author: str | None = None
    sensitivity: str | None = None
    record_kind: str | None = None
    project_statuses: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespaces", tuple(self.namespaces or ()))
        object.__setattr__(self, "project_statuses", tuple(self.project_statuses or ()))


def _q(alias: str, column: str) -> str:
    return f"{alias}.{column}" if alias else column


def _escape_like_prefix(prefix: str) -> str:
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def build_eligibility_where(
    retrieval_filter: RetrievalFilter | None = None,
    *,
    alias: str = "m",
) -> tuple[str, list[Any]]:
    filt = retrieval_filter or RetrievalFilter()
    if not filt.owner:
        raise ValueError("retrieval owner is required")

    clauses = [
        f"{_q(alias, 'archived')} = 0",
    ]
    params: list[Any] = [filt.owner]

    clauses.append(f"{_q(alias, 'owner')} = ?")

    namespaces = list(filt.namespaces)
    if namespaces:
        placeholders = ", ".join("?" for _ in namespaces)
        clauses.append(f"{_q(alias, 'namespace')} IN ({placeholders})")
        params.extend(namespaces)
    if filt.project_slug is not None:
        clauses.append(f"{_q(alias, 'namespace')} = ?")
        params.append(f"project:{filt.project_slug}")

    if filt.path_prefix is not None:
        clauses.append(f"{_q(alias, 'source_file')} LIKE ? ESCAPE '\\'")
        params.append(f"{_escape_like_prefix(filt.path_prefix)}%")

    exact_fields = {
        "source_id": filt.source_id,
        "author": filt.author,
        "sensitivity": filt.sensitivity,
        "record_kind": filt.record_kind,
    }
    for column, value in exact_fields.items():
        if value is not None:
            clauses.append(f"{_q(alias, column)} = ?")
            params.append(value)

    if filt.project_statuses:
        allowed_statuses = tuple(
            status for status in filt.project_statuses if status in ELIGIBLE_PROJECT_STATUSES
        )
        placeholders = ", ".join("?" for _ in allowed_statuses)
        if allowed_statuses:
            clauses.append(f"{_q(alias, 'project_status')} IN ({placeholders})")
            params.extend(allowed_statuses)
        else:
            clauses.append("1 = 0")
    else:
        placeholders = ", ".join("?" for _ in ELIGIBLE_PROJECT_STATUSES)
        clauses.append(
            f"({_q(alias, 'project_status')} IS NULL "
            f"OR {_q(alias, 'project_status')} IN ({placeholders}))"
        )
        params.extend(ELIGIBLE_PROJECT_STATUSES)

    return " AND ".join(clauses), params


def filter_eligible_ids(
    conn: sqlite3.Connection,
    candidate_ids: list[int] | set[int] | tuple[int, ...],
    retrieval_filter: RetrievalFilter | None = None,
) -> set[int]:
    ids = list(dict.fromkeys(candidate_ids))
    if not ids:
        return set()
    eligible: set[int] = set()
    where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT m.id FROM memories m WHERE m.id IN ({placeholders}) AND {where_sql}",
            (*batch, *params),
        ).fetchall()
        eligible.update(row[0] if isinstance(row, tuple) else row["id"] for row in rows)
    return eligible


def list_eligible_ids(
    conn: sqlite3.Connection,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[int]:
    where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
    rows = conn.execute(
        f"SELECT m.id FROM memories m WHERE {where_sql} ORDER BY m.id",
        params,
    ).fetchall()
    return [row[0] if isinstance(row, tuple) else row["id"] for row in rows]


def add_provenance(result: dict) -> dict:
    result.setdefault("source_file", None)
    result.setdefault("path", result.get("source_file"))
    result.setdefault("source_path", result.get("source_file"))
    for key in (
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
    ):
        result.setdefault(key, None)
    return result


def add_provenance_to_results(results: list[dict]) -> list[dict]:
    return [add_provenance(result) for result in results]
