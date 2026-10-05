"""
memory-v3 -- Retrieval Subpackage
3-stage search pipeline: coarse filtering, fine scoring, organization.

Public API
----------
search(conn, query, query_embedding, limit=10, content_type=None, config=None)
    Full 3-stage retrieval pipeline with lightweight organization.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from . import stage1_coarse, stage2_fine, stage3_organize

if TYPE_CHECKING:
    from ..config import Config
    from .filters import RetrievalFilter

logger = logging.getLogger(__name__)

__all__ = [
    "search",
    "stage1_coarse",
    "stage2_fine",
    "stage3_organize",
]


def search(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    limit: int = 10,
    content_type: str | None = None,
    config: Config | None = None,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[dict]:
    """
    Full 3-stage retrieval pipeline.

    Stage 1 (Coarse): Fast candidate filtering via communities + vec + FTS5.
    Stage 2 (Fine):   Deterministic RRF fusion of lexical/vector ranks.
    Stage 3 (Organize): Eligibility filtering and structure hints.

    Falls back to v2-style direct hybrid search when three-stage mode is disabled.

    Args:
        conn: SQLite connection with full memory-v3 schema.
        query: The raw user query string.
        query_embedding: Pre-computed embedding vector for the query.
        limit: Maximum number of results to return.
        content_type: Optional filter by content_type (fact, episode, etc.).
        config: Optional Config override. Uses get_config() if None.

    Returns:
        List of scored, organized memory dicts.
    """
    from ..config import get_config

    cfg = config or get_config()

    if getattr(cfg, "enable_three_stage", True):
        # Stage 1: Coarse filter
        candidates = stage1_coarse.get_candidates(
            conn,
            query,
            query_embedding,
            cfg,
            retrieval_filter,
        )

        if not candidates:
            logger.debug("Stage 1 returned no candidates; falling back to direct search")
            return _fallback_search(
                conn,
                query,
                query_embedding,
                limit,
                content_type,
                retrieval_filter,
            )

        # Stage 2: Fine scoring on candidates
        scored = stage2_fine.score_candidates(
            conn,
            candidates,
            query,
            query_embedding,
            cfg,
            retrieval_filter,
        )

        # Apply content_type filter if requested
        if content_type:
            scored = [r for r in scored if r.get("content_type") == content_type]

        # Stage 3: Organize results
        organized = stage3_organize.organize(
            conn,
            scored[:limit],
            query,
            cfg,
            retrieval_filter,
        )
        return organized

    else:
        # Fallback: v2-style direct search
        return _fallback_search(
            conn,
            query,
            query_embedding,
            limit,
            content_type,
            retrieval_filter,
        )


def _fallback_search(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    limit: int,
    content_type: str | None,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[dict]:
    """v2-compatible direct hybrid search without legacy scoring imports."""
    from .. import db

    results = db.hybrid_search(
        conn,
        query,
        query_embedding,
        limit=limit,
        content_type=content_type,
        retrieval_filter=retrieval_filter,
    )
    for result in results:
        result["final_score"] = result.get("hybrid_score", 0.0)
    return sorted(
        results,
        key=lambda result: (
            -result.get("final_score", 0.0),
            -result.get("bm25_score", 0.0),
            -result.get("vec_score", 0.0),
            result.get("id", 0),
        ),
    )[:limit]
