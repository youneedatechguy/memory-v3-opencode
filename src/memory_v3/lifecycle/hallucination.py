"""
memory-v3 -- HaluMem-Inspired Memory Validation

Verifies memory integrity by checking source files, detecting staleness,
and decaying confidence of unverified memories over time.

Verification statuses:
  - unverified: never checked
  - verified: source confirmed intact
  - stale: source file has changed since indexing
  - contradicted: source no longer matches memory content
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import Config, get_config
from ..db import update_memory
from ..source_policy import classify_source_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: str) -> str | None:
    """Compute SHA-256 hash of a file's content. Returns None if file missing."""
    try:
        content = Path(path).read_bytes()
        return hashlib.sha256(content).hexdigest()
    except OSError:
        return None


def _source_path_for_read(source_file: str, config: Config) -> tuple[Path | None, str | None]:
    if not config.vault_path:
        return None, "No vault path configured"

    source = source_file[3:] if source_file.startswith("kg:") else source_file
    raw_path = Path(source).expanduser()
    if ".." in raw_path.parts:
        return None, "Source path is outside approved source policy"

    root = Path(config.vault_path).expanduser().resolve(strict=True)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError:
        return None, "Source path is outside approved source policy"

    if not candidate.exists():
        return None, "Source file no longer exists"

    record = classify_source_path(candidate, config=config)
    if record is None:
        return None, "Source path is outside approved source policy"
    return root / record.relative_path, None


def verify_memory(conn: sqlite3.Connection, memory_id: int) -> dict:
    """Verify a single memory's source and confidence.

    Checks:
      1. Does the source_file exist?
      2. Does the source_hash match the current file hash?
      3. Updates verification_status accordingly.

    Returns: {memory_id, status, previous_status, source_file, reason}
    """
    row = conn.execute(
        "SELECT id, content, source_file, source_hash, verification_status, confidence "
        "FROM memories WHERE id = ? AND archived = 0",
        (memory_id,),
    ).fetchone()

    if row is None:
        return {
            "memory_id": memory_id,
            "status": "not_found",
            "reason": "Memory not found or archived",
        }

    mem = dict(row)
    source_file = mem.get("source_file")
    stored_hash = mem.get("source_hash")
    previous_status = mem.get("verification_status", "unverified")

    result = {
        "memory_id": memory_id,
        "previous_status": previous_status,
        "source_file": source_file,
    }

    # No source file -- cannot verify, but not contradicted
    if not source_file:
        new_status = "unverified"
        result["status"] = new_status
        result["reason"] = "No source file associated"
        update_memory(
            conn, memory_id,
            verification_status=new_status,
        )
        # Still update last_verified_at to record we checked
        conn.execute(
            "UPDATE memories SET last_verified_at = ? WHERE id = ?",
            (_now_iso(), memory_id),
        )
        conn.commit()
        return result

    cfg = get_config()
    source_path, reject_reason = _source_path_for_read(source_file, cfg)
    if source_path is None:
        new_status = "stale"
        result["status"] = new_status
        result["reason"] = reject_reason
        update_memory(conn, memory_id, verification_status=new_status)
        conn.execute(
            "UPDATE memories SET last_verified_at = ? WHERE id = ?",
            (_now_iso(), memory_id),
        )
        conn.commit()
        return result

    # Check if source file exists
    current_hash = _file_hash(str(source_path))
    if current_hash is None:
        new_status = "stale"
        result["status"] = new_status
        result["reason"] = "Source file no longer exists"
        update_memory(conn, memory_id, verification_status=new_status)
        conn.execute(
            "UPDATE memories SET last_verified_at = ? WHERE id = ?",
            (_now_iso(), memory_id),
        )
        conn.commit()
        return result

    # Compare hashes
    if stored_hash and stored_hash == current_hash:
        new_status = "verified"
        result["status"] = new_status
        result["reason"] = "Source hash matches"
    elif stored_hash and stored_hash != current_hash:
        new_status = "stale"
        result["status"] = new_status
        result["reason"] = "Source file has changed since indexing"
    else:
        # No stored hash -- store current and mark verified
        new_status = "verified"
        result["status"] = new_status
        result["reason"] = "Source hash recorded for first time"
        conn.execute(
            "UPDATE memories SET source_hash = ? WHERE id = ?",
            (current_hash, memory_id),
        )

    update_memory(conn, memory_id, verification_status=new_status)
    conn.execute(
        "UPDATE memories SET last_verified_at = ? WHERE id = ?",
        (_now_iso(), memory_id),
    )
    conn.commit()
    return result


def batch_verify(
    conn: sqlite3.Connection,
    memory_ids: list[int] | None = None,
    max_check: int = 50,
) -> dict:
    """Verify multiple memories.

    If memory_ids is None, checks the oldest unverified memories.

    Returns: {checked, verified, stale, unverified, errors, details}
    """
    stats = {
        "checked": 0,
        "verified": 0,
        "stale": 0,
        "unverified": 0,
        "errors": 0,
        "details": [],
    }

    if memory_ids is None:
        # Get oldest unverified or never-verified memories
        rows = conn.execute(
            """SELECT id FROM memories
            WHERE archived = 0
              AND (verification_status = 'unverified'
                   OR last_verified_at IS NULL)
            ORDER BY created_at ASC
            LIMIT ?""",
            (max_check,),
        ).fetchall()
        memory_ids = [row[0] if not isinstance(row, dict) else row["id"] for row in rows]
    else:
        memory_ids = memory_ids[:max_check]

    for mid in memory_ids:
        try:
            result = verify_memory(conn, mid)
            stats["checked"] += 1
            status = result.get("status", "unverified")
            if status == "verified":
                stats["verified"] += 1
            elif status == "stale":
                stats["stale"] += 1
            elif status == "not_found":
                stats["errors"] += 1
            else:
                stats["unverified"] += 1
            stats["details"].append(result)
        except Exception as e:
            stats["errors"] += 1
            stats["details"].append({
                "memory_id": mid,
                "status": "error",
                "reason": str(e),
            })

    return stats


def decay_confidence(
    conn: sqlite3.Connection,
    config: Config | None = None,
) -> dict:
    """Degrade confidence of unverified memories older than CONFIDENCE_DECAY_DAYS.

    For each qualifying memory:
      new_confidence = max(confidence - decay_rate, confidence_floor)

    Returns: {memories_decayed, total_confidence_lost}
    """
    cfg = config or get_config()
    decay_days = cfg.halumem_confidence_decay_days
    decay_rate = cfg.halumem_confidence_decay_rate

    cutoff = (datetime.now(timezone.utc) - timedelta(days=decay_days)).isoformat()

    # Find unverified memories older than the cutoff
    rows = conn.execute(
        """SELECT id, confidence, confidence_floor, governance_layer
        FROM memories
        WHERE archived = 0
          AND verification_status IN ('unverified', 'stale')
          AND created_at < ?
          AND protected = 0""",
        (cutoff,),
    ).fetchall()

    stats = {"memories_decayed": 0, "total_confidence_lost": 0.0}

    for row in rows:
        mem = dict(row)
        mid = mem["id"]
        current_conf = mem["confidence"]
        floor = mem.get("confidence_floor", 0.0) or 0.0
        gov_layer = mem.get("governance_layer", 3)

        # Never decay constitutional or legislative layers
        if gov_layer in (1, 2):
            continue

        new_conf = max(current_conf - decay_rate, floor)
        if new_conf < current_conf:
            conn.execute(
                "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
                (new_conf, _now_iso(), mid),
            )
            stats["memories_decayed"] += 1
            stats["total_confidence_lost"] += (current_conf - new_conf)

    conn.commit()
    return stats


def detect_staleness(
    conn: sqlite3.Connection,
    config: Config | None = None,
) -> list[dict]:
    """Find memories whose source files have been modified since indexing.

    Joins memories with file_hashes to check for hash mismatches,
    then also checks if the current on-disk file differs.

    Returns list of {memory_id, source_file, stored_hash, current_hash}.
    """
    stale = []
    cfg = config or get_config()

    # Method 1: Check against file_hashes table
    rows = conn.execute(
        """SELECT m.id, m.source_file, m.source_hash, fh.content_hash
        FROM memories m
        JOIN file_hashes fh ON m.source_file = fh.file_path
        WHERE m.archived = 0
          AND m.source_file IS NOT NULL
          AND m.source_hash IS NOT NULL
          AND m.source_hash != fh.content_hash"""
    ).fetchall()

    seen_ids = set()
    for row in rows:
        mem = dict(row)
        stale.append({
            "memory_id": mem["id"],
            "source_file": mem["source_file"],
            "stored_hash": mem["source_hash"],
            "current_hash": mem["content_hash"],
            "detection": "file_hashes_mismatch",
        })
        seen_ids.add(mem["id"])

    # Method 2: Check on-disk files directly for memories not in file_hashes
    rows2 = conn.execute(
        """SELECT m.id, m.source_file, m.source_hash
        FROM memories m
        LEFT JOIN file_hashes fh ON m.source_file = fh.file_path
        WHERE m.archived = 0
          AND m.source_file IS NOT NULL
          AND m.source_hash IS NOT NULL
          AND fh.file_path IS NULL"""
    ).fetchall()

    for row in rows2:
        mem = dict(row)
        mid = mem["id"]
        if mid in seen_ids:
            continue

        source_path, _ = _source_path_for_read(mem["source_file"], cfg)
        if source_path is None:
            continue
        current_hash = _file_hash(str(source_path))
        if current_hash is not None and current_hash != mem["source_hash"]:
            stale.append({
                "memory_id": mid,
                "source_file": mem["source_file"],
                "stored_hash": mem["source_hash"],
                "current_hash": current_hash,
                "detection": "disk_check",
            })

    return stale
