"""Deprecated lifecycle consolidation compatibility entry points."""

from __future__ import annotations

_DISABLED_RESULT = {
    "status": "disabled",
    "deprecated": True,
    "code": "lifecycle_consolidation_disabled",
    "error": "Lifecycle consolidation is disabled; active memory-v3 operation is graph-free.",
}


def _disabled_stats(cycle_type: str | None = None) -> dict:
    result = dict(_DISABLED_RESULT)
    if cycle_type is not None:
        result["cycle_type"] = cycle_type
    return result


def run_consolidation(
    conn: object,
    cycle_type: str = "full",
    config: object | None = None,
) -> dict:
    return _disabled_stats(cycle_type)


def _replay(conn: object, config: object) -> dict:
    return _disabled_stats("replay")


def _reorganize(conn: object, config: object) -> dict:
    return _disabled_stats("reorganize")


def _compress(conn: object, config: object) -> dict:
    return _disabled_stats("compress")


def _prune(conn: object, config: object) -> dict:
    return _disabled_stats("prune")
