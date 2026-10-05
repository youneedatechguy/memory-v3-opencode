"""
memory-v3 -- Stage 3: Post-Retrieval Organization
Applies eligibility filtering and lightweight structure hints for LLM consumption.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from .filters import RetrievalFilter

logger = logging.getLogger(__name__)


def _dominant_structure_type(results: list[dict]) -> str:
    """
    Determine the dominant structure_type among results by majority vote.

    Falls back to "narrative" if no clear winner.
    """
    if not results:
        return "narrative"

    counts: Counter = Counter()
    for r in results:
        st = r.get("structure_type", "narrative")
        counts[st] += 1

    best, best_count = counts.most_common(1)[0]
    # Only use the detected type if it covers at least 40% of results
    if best_count >= len(results) * 0.4:
        return best
    return "narrative"


def organize(
    conn: sqlite3.Connection,
    results: list[dict],
    query: str,
    config: Config,
    retrieval_filter: RetrievalFilter | None = None,
) -> list[dict]:
    """
    Stage 3: Post-retrieval organization and annotation.

    1. Detect dominant structure type among results
    2. Re-apply retrieval eligibility to scored rows
    3. Apply simple structure-appropriate ordering
    4. Generate a structure_hint for LLM consumption

    Args:
        conn: SQLite connection.
        results: Scored results from Stage 2.
        query: The raw user query string.
        config: The memory-v3 Config object.

    Returns:
        Organized and annotated result list.
    """
    if not results:
        return []

    if retrieval_filter is not None:
        from .filters import filter_eligible_ids

        eligible = filter_eligible_ids(
            conn,
            [r["id"] for r in results if isinstance(r.get("id"), int)],
            retrieval_filter,
        )
        results = [r for r in results if r.get("id") in eligible]
        if not results:
            return []

    # Determine dominant structure type
    enable_tags = getattr(config, "enable_structure_tags", True)
    if enable_tags:
        structure_type = _dominant_structure_type(results)
    else:
        structure_type = "narrative"

    layers: list[str] = []

    # Generate structure hint
    from .structure_tags import get_structure_hint

    hint = get_structure_hint(structure_type, len(results))

    # Attach metadata to each result
    for r in results:
        r["structure_hint"] = hint
        r["detected_structure"] = structure_type
        r["routed_layers"] = layers
        r["related_nodes"] = []
        r["see_also"] = []
        r["linked_memories"] = []

    logger.debug(
        "Stage 3 organized %d results (structure=%s, layers=%s)",
        len(results),
        structure_type,
        layers,
    )

    return results
