"""
memory-v3 -- Heuristic + RL-Ready Memory Action Decisions

Determines ADD / UPDATE / DELETE / NONE for incoming facts.
Supports two modes:
  - Heuristic: cosine threshold + LLM fallback (default)
  - RL: prompt-based simulation using action history (placeholder, flag-gated)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..config import Config, get_config
from ..db import log_action
from ..scoring.actr import cosine_similarity

# ---------------------------------------------------------------------------
# LLM prompt for ambiguous cases
# ---------------------------------------------------------------------------

_LLM_ACTION_PROMPT = """Given an existing memory and a new fact, decide the action.

Existing memory (ID {memory_id}):
"{existing}"

New fact:
"{new_fact}"

Choose ONE action and explain briefly:
- UPDATE: if the new fact updates/replaces the existing memory
- DELETE: if the new fact contradicts and invalidates the existing memory
- NONE: if the existing memory is still valid and the new fact adds nothing

Respond in JSON: {{"action": "UPDATE|DELETE|NONE", "reason": "..."}}"""

_RL_ACTION_PROMPT = """You are a memory management agent. Based on past action outcomes,
decide the best action for a new fact.

New fact:
"{new_fact}"

Similar existing memories:
{similar_list}

Recent action history (action -> outcome):
{action_history}

Choose ONE action with reasoning:
- ADD: The fact is genuinely new information worth storing
- UPDATE <id>: The fact updates an existing memory (specify which ID)
- DELETE <id>: The fact contradicts an existing memory (specify which ID)
- NONE: The fact is redundant or not worth storing

Respond in JSON: {{"action": "ADD|UPDATE|DELETE|NONE", "memory_id": <id or null>, "reason": "...", "confidence": <0.0-1.0>}}"""


class ActionDecider:
    """Heuristic + RL-ready action decision engine."""

    def __init__(self, conn: sqlite3.Connection, config: Optional[Config] = None):
        self.conn = conn
        self.cfg = config or get_config()

    def decide(
        self,
        new_fact: str,
        new_embedding: list[float] | np.ndarray,
        similar_memories: list[dict],
    ) -> dict:
        """Decide ADD/UPDATE/DELETE/NONE for a new fact.

        Parameters
        ----------
        new_fact : str
            The extracted fact to evaluate.
        new_embedding : array-like
            Embedding vector for the new fact.
        similar_memories : list[dict]
            List of dicts with keys: id, content, confidence, similarity.

        Returns
        -------
        dict with keys: action, memory_id, reason, confidence
        """
        if self.cfg.enable_rl_decider:
            return self._rl_decide(new_fact, new_embedding, similar_memories)
        return self._heuristic_decide(new_fact, new_embedding, similar_memories)

    def _heuristic_decide(
        self,
        new_fact: str,
        new_embedding: list[float] | np.ndarray,
        similar_memories: list[dict],
    ) -> dict:
        """Heuristic decision: cosine thresholds + LLM fallback.

        Thresholds:
          > 0.92 = NONE (too similar, skip)
          > 0.75 = LLM decides UPDATE vs NONE
          else   = ADD
        """
        if not similar_memories:
            result = {"action": "ADD", "memory_id": None, "reason": "No similar memories", "confidence": 0.9}
            self._log(new_fact, [], result)
            return result

        # Sort by similarity descending
        sorted_mems = sorted(similar_memories, key=lambda m: m["similarity"], reverse=True)
        top = sorted_mems[0]
        top_sim = top["similarity"]

        # High similarity -- almost identical, skip
        if top_sim > 0.92:
            result = {
                "action": "NONE",
                "memory_id": top["id"],
                "reason": f"Near-identical (cosine={top_sim:.3f})",
                "confidence": top_sim,
            }
            self._log(new_fact, [m["id"] for m in sorted_mems[:5]], result)
            return result

        # Medium similarity -- ask LLM
        if top_sim > 0.75:
            result = self._llm_decide(new_fact, top)
            self._log(new_fact, [m["id"] for m in sorted_mems[:5]], result)
            return result

        # Low similarity -- new information
        result = {
            "action": "ADD",
            "memory_id": None,
            "reason": f"Novel content (max cosine={top_sim:.3f})",
            "confidence": 0.85,
        }
        self._log(new_fact, [m["id"] for m in sorted_mems[:5]], result)
        return result

    def _rl_decide(
        self,
        new_fact: str,
        new_embedding: list[float] | np.ndarray,
        similar_memories: list[dict],
    ) -> dict:
        """RL-trained decision using prompt-based simulation.

        Builds a prompt with the new fact, similar memories, and recent
        action history, then asks the LLM to decide based on past outcomes.
        """
        model = self.cfg.llm_model

        # Build similar memories list for the prompt
        similar_list = ""
        for mem in similar_memories[:5]:
            similar_list += f"  - ID {mem['id']} (cosine={mem['similarity']:.3f}): \"{mem['content'][:150]}\"\n"

        # Get recent action history with outcomes
        action_history = self._get_recent_action_history(limit=10)
        history_str = ""
        for entry in action_history:
            history_str += f"  - {entry['action']} (confidence={entry['confidence']:.2f}): {entry['fact'][:80]}\n"

        if not history_str:
            history_str = "  (no prior history)\n"

        prompt = _RL_ACTION_PROMPT.format(
            new_fact=new_fact,
            similar_list=similar_list or "  (none)\n",
            action_history=history_str,
        )

        try:
            import ollama

            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 512},
            )
            content = response["message"]["content"].strip()
            parsed = self._parse_json_response(content)

            action = parsed.get("action", "NONE").upper()
            if action not in ("ADD", "UPDATE", "DELETE", "NONE"):
                action = "NONE"

            result = {
                "action": action,
                "memory_id": parsed.get("memory_id"),
                "reason": parsed.get("reason", "RL decision"),
                "confidence": float(parsed.get("confidence", 0.7)),
            }
        except Exception as e:
            # Fall back to heuristic on error
            result = self._heuristic_decide(new_fact, new_embedding, similar_memories)
            result["reason"] = f"RL fallback to heuristic: {e}"

        self._log(new_fact, [m["id"] for m in similar_memories[:5]], result, model_version="rl-v1")
        return result

    def record_outcome(self, action_log_id: int, outcome: str) -> bool:
        """Record user feedback for RL training.

        Stores the outcome (positive/negative/neutral) against
        the action log entry for future training signal.

        Returns True if the update succeeded.
        """
        try:
            # We store outcome in the model_version field as JSON annotation
            row = self.conn.execute(
                "SELECT model_version FROM action_log WHERE id = ?",
                (action_log_id,),
            ).fetchone()

            if row is None:
                return False

            existing = row[0] or ""
            # Append outcome data
            annotation = json.dumps({
                "original_model": existing,
                "outcome": outcome,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })

            self.conn.execute(
                "UPDATE action_log SET model_version = ? WHERE id = ?",
                (annotation, action_log_id),
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _llm_decide(self, new_fact: str, existing_mem: dict) -> dict:
        """Ask the LLM to decide between UPDATE, DELETE, and NONE."""
        model = self.cfg.llm_model
        prompt = _LLM_ACTION_PROMPT.format(
            memory_id=existing_mem["id"],
            existing=existing_mem["content"][:500],
            new_fact=new_fact,
        )

        try:
            import ollama

            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 256},
            )
            content = response["message"]["content"].strip()
            parsed = self._parse_json_response(content)

            action = parsed.get("action", "NONE").upper()
            if action not in ("UPDATE", "DELETE", "NONE"):
                action = "NONE"

            return {
                "action": action,
                "memory_id": existing_mem["id"],
                "reason": parsed.get("reason", "LLM decision"),
                "confidence": float(existing_mem.get("similarity", 0.8)),
            }
        except Exception as e:
            return {
                "action": "NONE",
                "memory_id": existing_mem["id"],
                "reason": f"LLM error, defaulting to NONE: {e}",
                "confidence": 0.5,
            }

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from an LLM response, tolerating markdown fences."""
        # Strip markdown code fences
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        import re
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def _log(
        self,
        new_fact: str,
        similar_ids: list[int],
        result: dict,
        model_version: Optional[str] = None,
    ) -> int:
        """Log the action decision to the action_log table."""
        return log_action(
            self.conn,
            new_fact=new_fact,
            similar_ids=similar_ids,
            action=result["action"],
            confidence=result.get("confidence", 0.0),
            model_version=model_version or self.cfg.llm_model,
        )

    def _get_recent_action_history(self, limit: int = 10) -> list[dict]:
        """Get recent action log entries for RL context."""
        rows = self.conn.execute(
            "SELECT new_fact, action, confidence, model_version FROM action_log "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()

        history = []
        for row in rows:
            history.append({
                "fact": row[0] if not isinstance(row, dict) else row["new_fact"],
                "action": row[1] if not isinstance(row, dict) else row["action"],
                "confidence": row[2] if not isinstance(row, dict) else row["confidence"],
                "model_version": row[3] if not isinstance(row, dict) else row["model_version"],
            })
        return history
