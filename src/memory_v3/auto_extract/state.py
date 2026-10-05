from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AutoExtractRunState:
    fingerprint: str
    last_run_ts: str
    status: str


class AutoExtractState:
    def __init__(self, state_path: Path):
        self.state_path = state_path

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_atomic(self, data: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def get(
        self, trigger_type: str, fingerprint: str
    ) -> AutoExtractRunState | None:
        data = self._load()
        trigger_bucket = data.get(trigger_type) or {}
        entry = trigger_bucket.get(fingerprint)
        if not isinstance(entry, dict):
            return None
        if "last_run_ts" not in entry or "status" not in entry:
            return None
        return AutoExtractRunState(
            fingerprint=fingerprint,
            last_run_ts=str(entry["last_run_ts"]),
            status=str(entry["status"]),
        )

    def mark_done(self, trigger_type: str, fingerprint: str, status: str = "done") -> None:
        data = self._load()
        trigger_bucket = data.get(trigger_type)
        if not isinstance(trigger_bucket, dict):
            trigger_bucket = {}
            data[trigger_type] = trigger_bucket

        trigger_bucket[fingerprint] = {
            "last_run_ts": _now_iso(),
            "status": status,
        }
        self._save_atomic(data)
