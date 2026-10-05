"""
memory-v3 -- Stage 1: Coarse Candidate Filtering
Fast, broad retrieval to build a candidate pool for fine ranking.

Pipeline:
  1. Fast vector search (vec0)
  2. FTS5 keyword search
  3. Union candidate IDs, cap at STAGE1_MAX_CANDIDATES
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import Config
    from .filters import RetrievalFilter

from .filters import build_eligibility_where, list_eligible_ids

logger = logging.getLogger(__name__)


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to raw bytes for sqlite-vec."""
    return np.array(vec, dtype=np.float32).tobytes()


def _l2_distance(query: np.ndarray, embedding: bytes) -> float:
    vector = np.frombuffer(embedding, dtype=np.float32)
    return float(np.linalg.norm(query - vector))


def _escape_fts5_query(query: str) -> str:
    """
    Escape a raw query string for safe use in FTS5 MATCH.

    Wraps each word in double quotes to prevent FTS5 syntax errors
    from special characters (colons, hyphens, etc.).
    """
    words = query.split()
    if not words:
        return '""'
    escaped = " OR ".join(f'"{w}"' for w in words if w.strip())
    return escaped or '""'


def _get_vector_candidate_ids(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 200,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[int]:
    """
    Fast vector similarity search via the vec0 virtual table.

    Returns memory IDs ordered by ascending distance (most similar first).
    """
    try:
        query_vec = np.array(query_embedding, dtype=np.float32)
        candidates: list[tuple[int, float]] = []
        eligible_ids = list_eligible_ids(conn, retrieval_filter)
        batch_size = 500
        for i in range(0, len(eligible_ids), batch_size):
            batch = eligible_ids[i : i + batch_size]
            placeholders = ", ".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT id, embedding FROM memory_vec WHERE id IN ({placeholders})",
                batch,
            ).fetchall()
            candidates.extend(
                (
                    row[0] if isinstance(row, tuple) else row["id"],
                    _l2_distance(query_vec, row[1] if isinstance(row, tuple) else row["embedding"]),
                )
                for row in rows
            )
        candidates.sort(key=lambda item: (item[1], item[0]))
        return [mid for mid, _distance in candidates[:limit]]
    except Exception as exc:
        logger.debug("Vector search failed: %s", exc)
        return []


def _get_fts_candidate_ids(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 200,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[int]:
    """
    FTS5 keyword search for candidate memory IDs.

    Returns rowids ordered by FTS5 rank (best match first).
    """
    escaped = _escape_fts5_query(query)
    try:
        where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
        rows = conn.execute(
            "SELECT memory_fts.rowid, rank FROM memory_fts "
            "JOIN memories m ON m.id = memory_fts.rowid "
            f"WHERE memory_fts MATCH ? AND {where_sql} "
            "ORDER BY rank LIMIT ?",
            (escaped, *params, limit),
        ).fetchall()
        return [row[0] if isinstance(row, tuple) else row["rowid"] for row in rows]
    except Exception as exc:
        logger.debug("FTS5 search failed: %s", exc)
        return []


def get_candidates(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    config: Config,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[int]:
    """
    Stage 1: Build a broad candidate pool from multiple retrieval channels.

    Combines vector and keyword candidates into a single deduplicated list,
    capped at config.stage1_max_candidates.

    Args:
        conn: SQLite connection with memory tables.
        query: The raw user query string.
        query_embedding: Pre-computed embedding vector for the query.
        config: The memory-v3 Config object.

    Returns:
        List of unique memory IDs (ints), up to stage1_max_candidates.
    """
    max_candidates = getattr(config, "stage1_max_candidates", 500)

    # --- Channel 1: Vector similarity candidates ---
    vec_ids = _get_vector_candidate_ids(
        conn,
        query_embedding,
        limit=200,
        retrieval_filter=retrieval_filter,
    )

    # --- Channel 2: FTS5 keyword candidates ---
    fts_ids = _get_fts_candidate_ids(
        conn,
        query,
        limit=200,
        retrieval_filter=retrieval_filter,
    )

    # --- Union all channels, preserving rough ordering ---
    # Vector and FTS results are ordered by relevance.
    seen: set[int] = set()
    ordered: list[int] = []

    for mid in vec_ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    for mid in fts_ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    # Cap at max_candidates
    result = ordered[:max_candidates]

    logger.debug(
        "Stage 1 candidates: %d (vec=%d, fts=%d, deduped=%d)",
        len(result),
        len(vec_ids),
        len(fts_ids),
        len(ordered),
    )

    return result
