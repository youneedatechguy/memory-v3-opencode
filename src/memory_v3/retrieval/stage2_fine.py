"""memory-v3 -- Stage 2: deterministic lexical/vector ranking."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import Config
    from .filters import RetrievalFilter

from .filters import add_provenance, build_eligibility_where, list_eligible_ids

logger = logging.getLogger(__name__)


def _serialize_f32(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


def _l2_distance(query: np.ndarray, embedding: bytes) -> float:
    vector = np.frombuffer(embedding, dtype=np.float32)
    return float(np.linalg.norm(query - vector))


def _escape_fts5_query(query: str) -> str:
    words = query.split()
    if not words:
        return '""'
    return " OR ".join(f'"{w}"' for w in words if w.strip()) or '""'


def _fetch_memories(
    conn: sqlite3.Connection,
    candidate_ids: list[int],
    retrieval_filter: RetrievalFilter | None = None,
) -> dict[int, dict]:
    memories: dict[int, dict] = {}
    batch_size = 500

    for i in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[i : i + batch_size]
        placeholders = ",".join("?" for _ in batch)
        where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
        rows = conn.execute(
            f"SELECT m.* FROM memories m WHERE m.id IN ({placeholders}) AND {where_sql}",
            (*batch, *params),
        ).fetchall()
        for row in rows:
            mem = dict(row)
            if isinstance(mem.get("tags"), str):
                try:
                    mem["tags"] = json.loads(mem["tags"])
                except (json.JSONDecodeError, TypeError):
                    mem["tags"] = []
            if isinstance(mem.get("linked_memories"), str):
                try:
                    mem["linked_memories"] = json.loads(mem["linked_memories"])
                except (json.JSONDecodeError, TypeError):
                    mem["linked_memories"] = []
            memories[mem["id"]] = add_provenance(mem)

    return memories


def _compute_bm25_ranks(
    conn: sqlite3.Connection,
    query: str,
    candidate_ids: set[int],
    retrieval_filter: RetrievalFilter | None = None,
) -> dict[int, int]:
    escaped = _escape_fts5_query(query)
    ranks: dict[int, int] = {}
    try:
        where_sql, params = build_eligibility_where(retrieval_filter, alias="m")
        rows = conn.execute(
            "SELECT memory_fts.rowid, rank FROM memory_fts "
            "JOIN memories m ON m.id = memory_fts.rowid "
            f"WHERE memory_fts MATCH ? AND {where_sql} "
            "ORDER BY rank LIMIT 1000",
            (escaped, *params),
        ).fetchall()
        position = 0
        for row in rows:
            rowid = row[0] if isinstance(row, tuple) else row["rowid"]
            if rowid in candidate_ids:
                ranks[rowid] = position
                position += 1
    except Exception as exc:
        logger.debug("BM25 ranking failed: %s", exc)

    return ranks


def _compute_vector_ranks(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    candidate_ids: set[int],
    retrieval_filter: RetrievalFilter | None = None,
) -> dict[int, int]:
    ranks: dict[int, int] = {}
    try:
        query_vec = np.array(query_embedding, dtype=np.float32)
        eligible_ids = [
            mid for mid in list_eligible_ids(conn, retrieval_filter) if mid in candidate_ids
        ]
        candidates: list[tuple[int, float]] = []
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
                    _l2_distance(
                        query_vec,
                        row[1] if isinstance(row, tuple) else row["embedding"],
                    ),
                )
                for row in rows
            )
        candidates.sort(key=lambda item: (item[1], item[0]))
        for position, (mid, _distance) in enumerate(candidates):
            ranks[mid] = position
    except Exception as exc:
        logger.debug("Vector ranking failed: %s", exc)

    return ranks


def _rrf_fusion(
    bm25_ranks: dict[int, int],
    vec_ranks: dict[int, int],
    all_ids: set[int],
    k: int = 60,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    penalty_rank = 9999

    for mid in all_ids:
        bm25_r = bm25_ranks.get(mid, penalty_rank)
        vec_r = vec_ranks.get(mid, penalty_rank)
        scores[mid] = 1.0 / (k + bm25_r) + 1.0 / (k + vec_r)

    return scores


def score_candidates(
    conn: sqlite3.Connection,
    candidate_ids: list[int],
    query: str,
    query_embedding: list[float],
    config: Config,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[dict]:
    if not candidate_ids:
        return []

    memories = _fetch_memories(conn, candidate_ids, retrieval_filter)
    if not memories:
        return []

    candidate_set = set(memories.keys())
    k = getattr(config, "rrf_k", 60)

    bm25_ranks = _compute_bm25_ranks(conn, query, candidate_set, retrieval_filter)
    vec_ranks = _compute_vector_ranks(conn, query_embedding, candidate_set, retrieval_filter)

    rrf_scores = _rrf_fusion(bm25_ranks, vec_ranks, candidate_set, k=k)
    scored_results: list[dict] = []

    for mid, mem in memories.items():
        rrf = rrf_scores.get(mid, 0.0)
        mem["hybrid_score"] = rrf
        mem["bm25_rank"] = bm25_ranks.get(mid, 9999)
        mem["vec_rank"] = vec_ranks.get(mid, 9999)
        mem["rrf_score"] = rrf
        mem["final_score"] = rrf
        scored_results.append(mem)

    scored_results.sort(
        key=lambda r: (
            -r["final_score"],
            r["bm25_rank"],
            r["vec_rank"],
            r["id"],
        )
    )

    logger.debug(
        "Stage 2 scored %d candidates (top score=%.4f)",
        len(scored_results),
        scored_results[0]["final_score"] if scored_results else 0.0,
    )

    return scored_results
