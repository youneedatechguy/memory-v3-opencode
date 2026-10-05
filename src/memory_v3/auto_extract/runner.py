from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config, get_config
from ..db import init_db
from ..lifecycle.extraction import process_conversation
from .collectors import (
    collect_git_payload,
    collect_git_wip_payload,
    collect_vault_payloads,
    compute_git_wip_fingerprint,
    compute_vault_fingerprint,
    default_daily_fingerprint,
)
from .state import AutoExtractState


@dataclass(frozen=True)
class AutoExtractResult:
    trigger_type: str
    fingerprint: str
    collected_chars: int
    extracted: dict
    skipped: bool
    reason: str


class AutoExtractRunner:
    def __init__(
        self,
        *,
        conn=None,
        config: Config | None = None,
        state_path: Path | None = None,
    ):
        self.cfg = config or get_config()
        self._conn = conn
        base_dir = Path(self.cfg.db_path).expanduser().resolve().parent
        self.state = AutoExtractState(
            state_path or (base_dir / "auto_extract_state.json")
        )

    @property
    def conn(self):
        if self._conn is None:
            self._conn = init_db(self.cfg.db_path)
        return self._conn

    def run(
        self,
        *,
        trigger_type: str,
        fingerprint: str,
        dry_run: bool = False,
        force: bool = False,
        git_dir: str | None = None,
        work_tree: str | None = None,
        text_override: str | None = None,
        vault_dir: str | None = None,
        since_days: int = 1,
        max_vault_files: int = 50,
        max_chars_per_file: int = 40000,
    ) -> AutoExtractResult:
        if text_override is not None and not self.cfg.direct_writes_enabled:
            return AutoExtractResult(
                trigger_type=trigger_type,
                fingerprint=fingerprint,
                collected_chars=0,
                extracted={},
                skipped=True,
                reason="Direct writes are disabled",
            )

        if trigger_type != "vault" and not self.cfg.direct_writes_enabled:
            return AutoExtractResult(
                trigger_type=trigger_type,
                fingerprint=fingerprint,
                collected_chars=0,
                extracted={},
                skipped=True,
                reason="Direct writes are disabled",
            )

        if trigger_type == "git":
            collected = collect_git_payload(
                fingerprint=fingerprint,
                git_dir=git_dir,
                work_tree=work_tree,
            )
        elif trigger_type == "git-wip":
            fp = fingerprint
            if fp == "auto":
                fp = compute_git_wip_fingerprint(
                    git_dir=git_dir,
                    work_tree=work_tree,
                )
            collected = collect_git_wip_payload(
                fingerprint=fp,
                git_dir=git_dir,
                work_tree=work_tree,
            )
            fingerprint = fp
        elif trigger_type == "daily":
            fp = fingerprint
            if fp == "auto":
                fp = default_daily_fingerprint()
            collected = collect_git_wip_payload(
                fingerprint=fp,
                git_dir=git_dir,
                work_tree=work_tree,
            )
            fingerprint = fp
        elif trigger_type == "vault":
            if not self.cfg.vault_auto_extract_enabled:
                return AutoExtractResult(
                    trigger_type=trigger_type,
                    fingerprint=fingerprint,
                    collected_chars=0,
                    extracted={},
                    skipped=True,
                    reason="Vault auto-extract is disabled",
                )

            fp_key = fingerprint
            if fp_key == "auto":
                fp_key = "since"

            state_entry = self.state.get(trigger_type, fp_key)
            if state_entry is None:
                since_dt = None
                if since_days >= 0:
                    from datetime import datetime, timedelta, timezone

                    since_dt = datetime.now(timezone.utc) - timedelta(days=since_days)
                else:
                    from datetime import datetime, timezone

                    since_dt = datetime.now(timezone.utc) - timedelta(days=1)
            else:
                from datetime import datetime

                since_dt = datetime.fromisoformat(state_entry.last_run_ts)

            vd = vault_dir or self.cfg.vault_path
            if not vd:
                return AutoExtractResult(
                    trigger_type=trigger_type,
                    fingerprint=fp_key,
                    collected_chars=0,
                    extracted={},
                    skipped=True,
                    reason="vault_dir is not set (MEMORY_V3_VAULT empty)",
                )

            fp = compute_vault_fingerprint(
                vault_dir=vd,
                since_dt=since_dt,
                max_files=max_vault_files,
                max_chars_per_file=max_chars_per_file,
                config=self.cfg,
            )
            vault_payloads = collect_vault_payloads(
                fingerprint=fp,
                vault_dir=vd,
                since_dt=since_dt,
                max_files=max_vault_files,
                max_chars_per_file=max_chars_per_file,
                config=self.cfg,
            )
            fingerprint = fp_key
            collected = None
        else:
            collected = None

        if trigger_type == "vault" and text_override is None:
            collected_text_chars = sum(len(payload.text) for payload in vault_payloads)
            if not vault_payloads or collected_text_chars == 0:
                return AutoExtractResult(
                    trigger_type=trigger_type,
                    fingerprint=fingerprint,
                    collected_chars=collected_text_chars,
                    extracted={},
                    skipped=True,
                    reason="Collected text is empty",
                )

            if dry_run:
                return AutoExtractResult(
                    trigger_type=trigger_type,
                    fingerprint=fingerprint,
                    collected_chars=collected_text_chars,
                    extracted={"dry_run": True},
                    skipped=False,
                    reason="Dry run; no extraction executed",
                )

            extracted = self._process_vault_payloads(vault_payloads)
            if extracted and extracted.get("errors", 0) == 0:
                self.state.mark_done(trigger_type, fingerprint, status="done")

            return AutoExtractResult(
                trigger_type=trigger_type,
                fingerprint=fingerprint,
                collected_chars=collected_text_chars,
                extracted=extracted,
                skipped=False,
                reason="",
            )

        if text_override is not None:
            text = text_override
            source = f"auto_extract:{trigger_type}"
            collected_text_chars = len(text)
        else:
            if collected is None:
                return AutoExtractResult(
                    trigger_type=trigger_type,
                    fingerprint=fingerprint,
                    collected_chars=0,
                    extracted={},
                    skipped=True,
                    reason=f"Unknown trigger_type '{trigger_type}'",
                )
            text = collected.text
            source = collected.source
            collected_text_chars = len(text)

        if trigger_type != "vault":
            state_entry = self.state.get(trigger_type, fingerprint)
            if state_entry is not None and not force:
                return AutoExtractResult(
                    trigger_type=trigger_type,
                    fingerprint=fingerprint,
                    collected_chars=collected_text_chars,
                    extracted={},
                    skipped=True,
                    reason=f"Already extracted (status={state_entry.status})",
                )

        if not text.strip():
            return AutoExtractResult(
                trigger_type=trigger_type,
                fingerprint=fingerprint,
                collected_chars=collected_text_chars,
                extracted={},
                skipped=True,
                reason="Collected text is empty",
            )

        if dry_run:
            return AutoExtractResult(
                trigger_type=trigger_type,
                fingerprint=fingerprint,
                collected_chars=collected_text_chars,
                extracted={"dry_run": True},
                skipped=False,
                reason="Dry run; no extraction executed",
            )

        extracted = process_conversation(
            self.conn,
            text=text,
            source=source,
            author="auto_extract",
            config=self.cfg,
        )

        status = "done" if extracted.get("errors", 0) == 0 else "error"
        if extracted:
            self.state.mark_done(trigger_type, fingerprint, status=status)

        return AutoExtractResult(
            trigger_type=trigger_type,
            fingerprint=fingerprint,
            collected_chars=collected_text_chars,
            extracted=extracted,
            skipped=False,
            reason="",
        )

    def _process_vault_payloads(self, payloads) -> dict:
        stats = {
            "facts_extracted": 0,
            "added": 0,
            "updated": 0,
            "deleted": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

        from ..source_policy import make_chunk_id, make_source_id

        for payload in payloads:
            record = payload.source_record
            source_id = make_source_id(record.owner, record.relative_path)
            result = process_conversation(
                self.conn,
                text=payload.text,
                source=record.relative_path,
                author="auto_extract",
                config=self.cfg,
                add_only=True,
                memory_metadata={
                    "owner": record.owner,
                    "namespace": record.namespace,
                    "sensitivity": "normal",
                    "record_kind": "derived",
                    "source_hash": payload.source_hash,
                    "source_id": source_id,
                    "chunk_id": make_chunk_id(source_id, payload.source_hash, 0),
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "source_updated_at": payload.source_updated_at,
                    "project_status": record.project_status,
                    "project_updated_at": record.project_updated,
                },
            )
            for key in ("facts_extracted", "added", "updated", "deleted", "skipped", "errors"):
                stats[key] += result.get(key, 0)
            for detail in result.get("details", []):
                detail.setdefault("source_file", record.relative_path)
                stats["details"].append(detail)

        return stats
