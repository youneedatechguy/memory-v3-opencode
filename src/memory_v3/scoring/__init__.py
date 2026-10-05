"""Scoring subpackage -- unified v3 scoring pipeline.

Combines ACT-R activation, FadeMem importance, Titans surprise,
fan effect penalties, and constitutional hierarchy into a single
``score_memory`` call that produces the final retrieval score.

Public API
----------
score_memory(memory, hybrid_score, context_tags, tag_fan, config=None)
    Compute the unified v3 final score for a single memory.

rank_results(conn, results, query, context_tags=None, config=None)
    Apply the full v3 scoring pipeline to a batch of search results
    and return them sorted by final score descending.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .actr import (
    activation_noise,
    base_level_activation,
    cosine_similarity,
    full_activation,
    retrieval_probability,
    spreading_activation,
)
from .fademem import decay_value, importance_score, should_archive
from .fan_effect import spreading_activation_with_fan
from .hierarchy import LAYER_CONFIGS, LayerConfig, get_layer_config

__all__ = [
    # actr
    "base_level_activation",
    "spreading_activation",
    "activation_noise",
    "full_activation",
    "retrieval_probability",
    "cosine_similarity",
    # fademem
    "importance_score",
    "decay_value",
    "should_archive",
    # surprise
    "compute_surprise",
    # fan_effect
    "spreading_activation_with_fan",
    # hierarchy
    "get_layer_config",
    "LAYER_CONFIGS",
    "LayerConfig",
    # pipeline
    "score_memory",
    "rank_results",
]


def __getattr__(name: str):
    if name == "compute_surprise":
        from .surprise import compute_surprise

        return compute_surprise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Lightweight config shim -- avoids hard dependency on ..config at import time
# ---------------------------------------------------------------------------


class _DefaultConfig:
    """Fallback configuration when the real config module is unavailable."""

    ENABLE_FAN_EFFECT: bool = True
    ENABLE_HALUMEM: bool = True
    FAN_DELTA: float = 0.3


def _get_config(config=None):
    if config is not None:
        return config
    try:
        from ..config import get_config

        return get_config()
    except (ImportError, AttributeError):
        return _DefaultConfig()


# ---------------------------------------------------------------------------
# Single-memory scoring
# ---------------------------------------------------------------------------


def score_memory(
    memory: dict[str, Any],
    hybrid_score: float,
    context_tags: set[str],
    tag_fan: dict[str, int],
    config=None,
) -> tuple[float, float, float]:
    """Compute the unified v3 final score for a memory.

    Returns (final_score, activation, confidence).

    Final score formula:
        final = 0.60 * hybrid_score + 0.25 * (A / 10.0) + 0.15 * confidence
    """
    cfg = _get_config(config)
    layer_config = get_layer_config(memory.get("governance_layer", 3))

    # Resolve memory tags
    memory_tags = set(memory.get("tags", []) or [])

    # ACT-R spreading activation (with or without fan penalty)
    if getattr(cfg, "ENABLE_FAN_EFFECT", True):
        S = spreading_activation_with_fan(
            memory_tags,
            context_tags,
            tag_fan,
            fan_delta=getattr(cfg, "FAN_DELTA", 0.3),
        )
    else:
        S = spreading_activation(memory_tags, context_tags, tag_fan)

    # Base-level activation
    B = base_level_activation(
        memory.get("access_count", 1),
        memory["created_at"],
        memory.get("last_accessed_at", memory["created_at"]),
    )

    noise = activation_noise()
    A = max(B + S + noise, layer_config.activation_floor)

    # Confidence discount (HaluMem verification-aware)
    conf = memory.get("confidence", 0.95)
    if getattr(cfg, "ENABLE_HALUMEM", True):
        status = memory.get("verification_status", "unverified")
        if status == "stale":
            conf *= 0.7
        elif status == "contradicted":
            conf *= 0.3

    # Final composite: 60% search relevance, 25% activation, 15% confidence
    final = 0.60 * hybrid_score + 0.25 * (A / 10.0) + 0.15 * conf
    return final, A, conf


# ---------------------------------------------------------------------------
# Batch re-ranking
# ---------------------------------------------------------------------------


def rank_results(
    conn,
    results: list[dict[str, Any]],
    query: str,
    context_tags: set[str] | None = None,
    config=None,
) -> list[dict[str, Any]]:
    """Apply the full v3 scoring pipeline to search results and re-rank.

    Each result dict is expected to have at least:
        - hybrid_score (float): raw BM25+vector score
        - tags (list[str]): memory tags
        - access_count (int)
        - created_at (datetime or ISO string)
        - last_accessed_at (datetime or ISO string)
        - governance_layer (int, optional)
        - confidence (float, optional)
        - verification_status (str, optional)

    Returns the results list sorted by final_score descending, with
    ``final_score``, ``activation``, and ``adj_confidence`` added to each dict.
    """
    if not results:
        return []

    context_tags = context_tags or set()
    cfg = _get_config(config)

    # Build tag fan counts across the result set
    tag_counter: Counter = Counter()
    for r in results:
        for tag in r.get("tags") or []:
            tag_counter[tag] += 1
    tag_fan: dict[str, int] = dict(tag_counter)

    # Score each result
    for r in results:
        final, activation, adj_conf = score_memory(
            r,
            r.get("hybrid_score", 0.0),
            context_tags,
            tag_fan,
            config=cfg,
        )
        r["final_score"] = final
        r["activation"] = activation
        r["adj_confidence"] = adj_conf

    # Sort descending by final score
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results
