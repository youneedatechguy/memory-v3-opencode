"""Active handoff sync helpers for the lean memory-v3 server."""

from datetime import datetime

from . import db


def sync_agent(agent_id: str, conn, now: datetime | str | None = None) -> dict:
    """Return active owner-scoped handoffs without reading legacy changelogs."""
    handoffs = db.list_active_handoffs(conn, agent_id=agent_id, now=now)
    return {
        "agent_id": agent_id,
        "deprecated": True,
        "mode": "active_handoffs",
        "compatibility_mode": False,
        "count": len(handoffs),
        "handoffs": handoffs,
    }


def get_unresolved_conflicts(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT * FROM conflicts
        WHERE resolved = 0
        ORDER BY created_at DESC"""
    ).fetchall()
    return [dict(r) for r in rows]
