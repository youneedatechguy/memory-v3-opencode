"""Lifecycle consolidation — 4-phase sleep cycle (replay, reorganize, compress, prune).

Re-enabled from disabled stubs. Uses existing FadeMem scoring and db CRUD.
Phases run against non-archived memories in the SQLite store.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from ..config import Config, get_config
from ..db import archive_memory, update_memory
from ..embeddings import embed_with_cache
from ..scoring.fademem import importance_score, decay_value, should_archive, get_memory_layer


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _val(row, key, default=None):
    """Read a column from a sqlite Row, dict, or tuple."""
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        return default


def run_consolidation(
    conn: sqlite3.Connection,
    cycle_type: str = "full",
    config: Optional[Config] = None,
) -> dict:
    """Run the 4-phase consolidation cycle.

    Phases:
      - replay: recompute importance from recency/frequency/surprise
      - reorganize: re-classify memory_layer (LTM/STM/current)
      - compress: merge near-duplicate memories
      - prune: archive low-importance, old memories per governance rules
    """
    cfg = config or get_config()
    results = {"replay": {}, "reorganize": {}, "compress": {}, "prune": {}}

    if cycle_type in ("replay", "full"):
        results["replay"] = _replay(conn, cfg)
    if cycle_type in ("reorganize", "full"):
        results["reorganize"] = _reorganize(conn, cfg)
    if cycle_type in ("compress", "full"):
        results["compress"] = _compress(conn, cfg)
    if cycle_type in ("prune", "full"):
        results["prune"] = _prune(conn, cfg)

    total_updated = (
        results["replay"].get("updated", 0)
        + results["reorganize"].get("updated", 0)
        + results["compress"].get("merged", 0)
    )

    return {
        "status": "ok",
        "deprecated": False,
        "cycle_type": cycle_type,
        "phases": results,
        "importance_updated": total_updated,
        "layers_changed": results["reorganize"].get("layers_changed", 0),
        "archived": results["prune"].get("archived", 0),
    }


def _replay(conn: sqlite3.Connection, cfg: Config) -> dict:
    """Re-activate memories: recompute importance from access patterns."""
    rows = conn.execute(
        """
        SELECT id, access_count, last_accessed_at, importance_score,
               surprise_score, decay_rate
        FROM memories
        WHERE archived = 0
        """
    ).fetchall()

    updated = 0
    now = datetime.now(timezone.utc)

    for row in rows:
        mid = _val(row, "id", row[0])
        access_count = _val(row, "access_count", 0) or 0
        last_accessed = _val(row, "last_accessed_at")
        old_importance = float(_val(row, "importance_score", 0.5) or 0.5)
        surprise = float(_val(row, "surprise_score", 0.5) or 0.5)
        decay_rate = float(_val(row, "decay_rate", 0.1) or 0.1)

        hours_since = 0.0
        if last_accessed:
            try:
                last_dt = datetime.fromisoformat(str(last_accessed))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                hours_since = max((now - last_dt).total_seconds() / 3600.0, 0.0)
            except Exception:
                hours_since = 0.0

        relevance = min(max(old_importance, 0.0), 1.0)
        max_ac = max(access_count + 1, 1)
        frequency = math.log(access_count + 1) / math.log(max_ac + 1)
        recency = math.exp(-decay_rate * hours_since)

        new_importance = (
            0.30 * relevance + 0.25 * frequency + 0.25 * recency + 0.20 * surprise
        )
        new_importance = max(min(new_importance, 1.0), 0.0)

        if abs(new_importance - old_importance) > 0.01:
            update_memory(conn, mid, importance_score=new_importance)
            updated += 1

    return {"updated": updated, "processed": len(rows)}


def _reorganize(conn: sqlite3.Connection, cfg: Config) -> dict:
    """Re-classify memory_layer (LTM/STM/current) based on current importance."""
    rows = conn.execute(
        """
        SELECT id, importance_score, memory_layer
        FROM memories
        WHERE archived = 0
        """
    ).fetchall()

    updated = 0
    layers_changed = 0

    for row in rows:
        mid = _val(row, "id", row[0])
        importance = float(_val(row, "importance_score", 0.5) or 0.5)
        old_layer = _val(row, "memory_layer", "STM")

        new_layer = get_memory_layer(importance)

        if new_layer != old_layer:
            update_memory(conn, mid, memory_layer=new_layer)
            layers_changed += 1
        updated += 1

    return {"updated": updated, "layers_changed": layers_changed}


def _compress(conn: sqlite3.Connection, cfg: Config) -> dict:
    """Merge near-duplicate memories (cosine >= merge_threshold)."""
    threshold = float(cfg.consolidation_merge_threshold or 0.85)

    rows = conn.execute(
        """
        SELECT id, content, importance_score
        FROM memories
        WHERE archived = 0
        ORDER BY importance_score DESC
        """
    ).fetchall()

    merged = 0
    seen_ids = set()

    for i, row_a in enumerate(rows):
        mid_a = _val(row_a, "id", row_a[0])
        if mid_a in seen_ids:
            continue
        seen_ids.add(mid_a)

        content_a = _val(row_a, "content", "") or ""
        imp_a = float(_val(row_a, "importance_score", 0.5) or 0.5)

        try:
            emb_a = embed_with_cache(content_a)
        except Exception:
            continue

        for row_b in rows[i + 1 :]:
            mid_b = _val(row_b, "id", row_b[0])
            if mid_b in seen_ids:
                continue

            content_b = _val(row_b, "content", "") or ""
            try:
                emb_b = embed_with_cache(content_b)
            except Exception:
                continue

            sim = _cosine(emb_a, emb_b)
            if sim >= threshold:
                imp_b = float(_val(row_b, "importance_score", 0.5) or 0.5)
                victim = mid_b if imp_b < imp_a else mid_a
                if archive_memory(conn, victim, reason="duplicate"):
                    seen_ids.add(victim)
                    if victim == mid_a:
                        break
                merged += 1

    return {"merged": merged, "processed": len(rows)}


def _cosine(a, b) -> float:
    import numpy as np

    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _prune(conn: sqlite3.Connection, cfg: Config) -> dict:
    """Archive low-importance, old memories per governance layer rules."""
    from ..scoring.hierarchy import classify_governance_layer

    rows = conn.execute(
        """
        SELECT id, importance_score, governance_layer, created_at, tags, protected
        FROM memories
        WHERE archived = 0
        """
    ).fetchall()

    archived = 0
    now = datetime.now(timezone.utc)

    for row in rows:
        mid = _val(row, "id", row[0])
        importance = float(_val(row, "importance_score", 0.5) or 0.5)
        gov_layer = int(_val(row, "governance_layer", 3) or 3)
        created_at = _val(row, "created_at")
        tags_raw = _val(row, "tags", "[]") or "[]"
        protected = bool(_val(row, "protected", 0))

        age_days = 0.0
        if created_at:
            try:
                created_dt = datetime.fromisoformat(str(created_at))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                age_days = (now - created_dt).total_seconds() / 86400.0
            except Exception:
                age_days = 0.0

        tags = set()
        try:
            tags = set(json.loads(tags_raw))
        except Exception:
            tags = set()

        if should_archive(
            importance=importance,
            age_days=age_days,
            tags=tags,
            governance_layer=gov_layer,
        ):
            if archive_memory(conn, mid, reason="decay"):
                archived += 1

    return {"archived": archived, "processed": len(rows)}
