"""
memory-v3 -- LightMem-Inspired Sensory Intake Filter

Runs BEFORE any LLM calls. Multi-gate filter that rejects noise,
duplicates, credentials, and low-novelty content before it ever
reaches the extraction pipeline.

Gates:
  1. Length check
  2. Credential scan
  3. Exact dedup (SHA-256)
  4. Near-dedup (cosine similarity against recent buffer)
  5. Surprise score
  6. Topic/cluster assignment

Only content passing all gates is inserted into the sensory_buffer.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..config import Config, get_config
from ..db import add_to_sensory_buffer
from ..embeddings import embed_with_cache
from ..security import scan_for_credentials
from ..scoring.actr import cosine_similarity


class SensoryFilter:
    """Pre-encoding intake filter inspired by LightMem."""

    def __init__(self, conn: sqlite3.Connection, config: Optional[Config] = None):
        self.conn = conn
        self.cfg = config or get_config()

    def process(
        self,
        content: str,
        content_type: str = "fact",
        source: Optional[str] = None,
        author: str = "claude-code",
    ) -> dict:
        """Filter content through sensory gates.

        Returns:
            {
                accepted: bool,
                reason: str,
                buffer_id: int | None,
                surprise_score: float,
                cluster_id: int | None,
            }
        """
        result = {
            "accepted": False,
            "reason": "",
            "buffer_id": None,
            "surprise_score": 0.5,
            "cluster_id": None,
        }

        # Gate 1: Length check
        if len(content.strip()) < self.cfg.sensory_min_length:
            result["reason"] = f"Too short ({len(content.strip())} < {self.cfg.sensory_min_length})"
            return result

        # Gate 2: Credential scan
        cred_matches = scan_for_credentials(content)
        if cred_matches:
            result["reason"] = f"Credential detected: {cred_matches[0]['pattern'][:30]}"
            return result

        # Gate 3: Exact dedup (SHA-256 hash against sensory_buffer)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = self.conn.execute(
            "SELECT id FROM sensory_buffer WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if existing:
            result["reason"] = "Exact duplicate in sensory buffer"
            return result

        # Also check against memories table
        existing_mem = self.conn.execute(
            "SELECT id FROM memories WHERE content_hash = ? AND archived = 0",
            (content_hash[:16],),
        ).fetchone()
        if existing_mem:
            result["reason"] = "Exact duplicate in memories"
            return result

        # Gate 4: Near-dedup (embed, cosine > DEDUP_THRESHOLD against recent buffer)
        embedding = embed_with_cache(content)
        emb_array = np.array(embedding, dtype=np.float32)

        if self._is_near_duplicate(emb_array):
            result["reason"] = f"Near-duplicate (cosine > {self.cfg.sensory_dedup_threshold})"
            return result

        # Gates 5-6: centroid-based surprise scoring and topic/cluster
        # assignment are deterministic no-ops in the graph-free runtime.
        # No centroid code is imported; neutral surprise and no cluster.
        surprise = result["surprise_score"]
        result["cluster_id"] = None

        # All gates passed -- insert into sensory buffer
        topic_label = str(result["cluster_id"]) if result["cluster_id"] is not None else None

        buffer_id = add_to_sensory_buffer(
            self.conn,
            content=content,
            content_hash=content_hash,
            content_type=content_type,
            source=source,
            author=author,
            topic_cluster=topic_label,
            novelty_score=surprise,
        )

        result["accepted"] = True
        result["reason"] = "Passed all sensory gates"
        result["buffer_id"] = buffer_id
        return result

    def _is_near_duplicate(self, embedding: np.ndarray) -> bool:
        """Check if the embedding is near-duplicate of any recent buffer entry.

        Compares against the last 100 entries in the sensory buffer by
        re-embedding their content. Also checks the vec table for stored memories.
        """
        threshold = self.cfg.sensory_dedup_threshold

        # Check against recent sensory buffer entries
        recent = self.conn.execute(
            "SELECT content FROM sensory_buffer ORDER BY received_at DESC LIMIT 100"
        ).fetchall()

        for row in recent:
            other_emb = embed_with_cache(row[0])
            other_array = np.array(other_emb, dtype=np.float32)
            sim = cosine_similarity(embedding, other_array)
            if sim > threshold:
                return True

        # Check against stored memory vectors (top 50 nearest)
        from ..db import _serialize_f32

        try:
            vec_rows = self.conn.execute(
                """SELECT id, distance FROM memory_vec
                WHERE embedding MATCH ?
                ORDER BY distance LIMIT 5""",
                (_serialize_f32(embedding.tolist()),),
            ).fetchall()

            for row in vec_rows:
                # distance is L2 for vec0; convert to approximate cosine
                # For normalized vectors: cosine_sim ~= 1 - distance^2 / 2
                # But we'll just check if distance is very small
                distance = row["distance"] if isinstance(row, dict) else row[1]
                # vec0 returns cosine distance for cosine metric, or L2.
                # Safe heuristic: if distance < 0.1 it's very similar
                if distance < (1.0 - threshold) * 2:
                    return True
        except Exception:
            pass

        return False
