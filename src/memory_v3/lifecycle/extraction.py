"""
memory-v3 -- 2-Pass LLM Extraction Pipeline

Pass 1: Extract discrete facts from conversation text.
Pass 2: For each fact, decide action (ADD / UPDATE / DELETE / NONE)
         based on cosine similarity with existing memories.

Uses Ollama local LLM for both passes. Model configurable via
MEMORY_V3_LLM_MODEL env var (default: qwen2.5:3b).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..config import Config, get_config
from ..db import add_memory, update_memory, archive_memory, _serialize_f32
from ..embeddings import embed_with_cache
from ..scoring.actr import cosine_similarity
from ..scoring.hierarchy import classify_governance_layer
from .sensory import SensoryFilter
from .action_decider import ActionDecider
from ..providers import get_llm

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Extract discrete facts from this conversation. Categories:
1. User preferences (likes, dislikes, habits, style)
2. Personal details (name, location, role, expertise, family)
3. Plans and goals (projects, deadlines, intentions)
4. Active projects (repos, tools, systems being built)
5. Technical knowledge (languages, frameworks, architecture)
6. Corrections (things the user explicitly corrected)
7. Emotional context (feelings, reactions, moods expressed)

Rules:
- One fact per line, prefixed with "- "
- Write each fact as a standalone statement
- Include the date context if mentioned: [YYYY-MM-DD]
- Skip greetings, filler, and meta-conversation
- If nothing extractable, respond with: NONE

Conversation:
{text}

Extracted facts:"""

ACTION_PROMPT = """Given an existing memory and a new fact, decide the action.

Existing memory (ID {memory_id}):
"{existing}"

New fact:
"{new_fact}"

Choose ONE action and explain briefly:
- UPDATE: if the new fact updates/replaces the existing memory
- DELETE: if the new fact contradicts and invalidates the existing memory
- NONE: if the existing memory is still valid and the new fact adds nothing

Respond in JSON: {{"action": "UPDATE|DELETE|NONE", "reason": "..."}}"""

# ---------------------------------------------------------------------------
# Emotion detection
# ---------------------------------------------------------------------------

EMOTION_WORDS = {
    "happy", "sad", "angry", "frustrated", "excited", "anxious", "worried",
    "proud", "disappointed", "grateful", "annoyed", "confused", "overwhelmed",
    "relieved", "nervous", "thrilled", "upset", "stressed", "hopeful",
    "satisfied", "exhausted", "motivated", "discouraged", "delighted",
    "furious", "ecstatic", "miserable", "content", "passionate",
}

EMOTION_PATTERNS = [
    re.compile(r"\bi\s+(?:feel|felt|am feeling|was feeling)\s+\w+", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am)\s+(?:so\s+)?(?:happy|sad|angry|frustrated|excited|tired|stressed)", re.IGNORECASE),
    re.compile(r"\bthis\s+(?:makes|made)\s+me\s+(?:feel\s+)?\w+", re.IGNORECASE),
    re.compile(r"\bi\s+(?:love|hate|can't stand|adore)\s+", re.IGNORECASE),
    re.compile(r"[!]{2,}", re.IGNORECASE),
]


def detect_emotion(text: str) -> bool:
    """Detect if text contains emotional content."""
    text_lower = text.lower()
    for word in EMOTION_WORDS:
        if word in text_lower:
            return True
    for pattern in EMOTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def detect_content_type(fact: str) -> str:
    """Classify a fact into a content type based on keyword patterns."""
    fact_lower = fact.lower()

    # Check for corrections
    if any(kw in fact_lower for kw in ["actually", "correction", "not ", "wrong", "corrected", "instead"]):
        if any(kw in fact_lower for kw in ["correction", "corrected", "actually no", "not true"]):
            return "correction"

    # Check for decisions
    if any(kw in fact_lower for kw in ["decided", "decision", "chose", "will use", "going with", "committed to"]):
        return "decision"

    # Check for identity
    if any(kw in fact_lower for kw in ["my name is", "i am ", "i'm ", "my role", "i work as"]):
        if any(kw in fact_lower for kw in ["name is", "i am a", "i'm a", "my role is"]):
            return "identity"

    # Check for person references
    if any(kw in fact_lower for kw in ["person:", "family:", "colleague:", "friend:", "partner:"]):
        return "person"

    # Check for emotional content
    if detect_emotion(fact):
        return "episode"

    # Check for project/technical
    if any(kw in fact_lower for kw in ["project", "repo", "codebase", "building", "developing"]):
        return "fact"

    # Check for preferences
    if any(kw in fact_lower for kw in ["prefer", "like", "dislike", "favorite", "always", "never", "hate"]):
        return "fact"

    # Check for plans
    if any(kw in fact_lower for kw in ["plan", "goal", "want to", "going to", "intend", "deadline"]):
        return "fact"

    return "fact"


# ---------------------------------------------------------------------------
# Pass 1: Extract facts
# ---------------------------------------------------------------------------

def extract_facts(
    text: str,
    date: Optional[str] = None,
    config: Optional[Config] = None,
) -> list[str]:
    """Extract discrete facts from text using the local LLM.

    Returns a list of fact strings.
    """
    cfg = config or get_config()

    # Inject date context if available
    date_str = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = EXTRACT_PROMPT.format(text=text)
    if date:
        prompt = f"[Context: today is {date_str}]\n\n{prompt}"

    try:
        content = get_llm().chat(prompt, max_tokens=2048)
    except Exception as e:
        return [f"[extraction error: {e}]"]

    if content.upper() == "NONE" or not content:
        return []

    # Parse bullet-point facts
    facts = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            fact = line[2:].strip()
            if fact and len(fact) > 10:
                facts.append(fact)
        elif line and len(line) > 10 and not line.startswith("#"):
            # Accept non-prefixed lines too if they look like facts
            facts.append(line)

    return facts


# ---------------------------------------------------------------------------
# Pass 2: Decide action for each fact
# ---------------------------------------------------------------------------

def decide_action(
    conn: sqlite3.Connection,
    new_fact: str,
    new_embedding: list[float],
    config: Optional[Config] = None,
) -> dict:
    """Decide ADD/UPDATE/DELETE/NONE for a new fact.

    Uses cosine similarity thresholds with LLM fallback:
      > 0.92 = NONE (too similar, skip)
      > 0.75 = ask LLM to decide UPDATE vs NONE
      else    = ADD (new information)

    Returns: {action, memory_id, reason, confidence}
    """
    cfg = config or get_config()
    emb_array = np.array(new_embedding, dtype=np.float32)

    # Find similar existing memories via vec search
    try:
        rows = conn.execute(
            """SELECT id, distance FROM memory_vec
            WHERE embedding MATCH ?
            ORDER BY distance LIMIT 10""",
            (_serialize_f32(new_embedding),),
        ).fetchall()
    except Exception:
        rows = []

    if not rows:
        return {"action": "ADD", "memory_id": None, "reason": "No similar memories found", "confidence": 0.9}

    # Build similar memories list with cosine similarities
    similar_memories = []
    for row in rows:
        mid = row[0] if not isinstance(row, dict) else row["id"]
        mem = conn.execute(
            "SELECT id, content, confidence FROM memories WHERE id = ? AND archived = 0",
            (mid,),
        ).fetchone()
        if mem:
            # Get the stored embedding for accurate cosine
            vec_row = conn.execute(
                "SELECT embedding FROM memory_vec WHERE id = ?", (mid,)
            ).fetchone()
            if vec_row:
                stored_emb = np.frombuffer(
                    vec_row[0] if not isinstance(vec_row, dict) else vec_row["embedding"],
                    dtype=np.float32,
                )
                sim = cosine_similarity(emb_array, stored_emb)
                similar_memories.append({
                    "id": mem[0] if not isinstance(mem, dict) else mem["id"],
                    "content": mem[1] if not isinstance(mem, dict) else mem["content"],
                    "confidence": mem[2] if not isinstance(mem, dict) else mem["confidence"],
                    "similarity": float(sim),
                })

    if not similar_memories:
        return {"action": "ADD", "memory_id": None, "reason": "No active similar memories", "confidence": 0.9}

    # Use ActionDecider for the decision
    decider = ActionDecider(conn, config=cfg)
    return decider.decide(new_fact, new_embedding, similar_memories)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_conversation(
    conn: sqlite3.Connection,
    text: str,
    source: Optional[str] = None,
    author: str = "claude-code",
    date: Optional[str] = None,
    config: Optional[Config] = None,
) -> dict:
    """Full 2-pass extraction pipeline.

    1. Run sensory filter (if enabled)
    2. Extract facts from text
    3. For each fact, decide action and execute

    Returns summary: {facts_extracted, added, updated, deleted, skipped, errors, details}
    """
    cfg = config or get_config()
    stats = {
        "facts_extracted": 0,
        "added": 0,
        "updated": 0,
        "deleted": 0,
        "skipped": 0,
        "errors": 0,
        "details": [],
    }

    # Pass 1: Extract facts
    facts = extract_facts(text, date=date, config=cfg)
    stats["facts_extracted"] = len(facts)

    if not facts:
        return stats

    # Process each fact
    for fact in facts:
        try:
            # Run sensory filter if enabled
            if cfg.enable_sensory:
                sensory = SensoryFilter(conn, config=cfg)
                gate_result = sensory.process(
                    content=fact,
                    content_type=detect_content_type(fact),
                    source=source,
                    author=author,
                )
                if not gate_result["accepted"]:
                    stats["skipped"] += 1
                    stats["details"].append({
                        "fact": fact[:100],
                        "action": "SKIPPED",
                        "reason": gate_result["reason"],
                    })
                    continue

            # Embed the fact
            embedding = embed_with_cache(fact)

            # Pass 2: Decide action
            decision = decide_action(conn, fact, embedding, config=cfg)
            action = decision["action"]

            if action == "ADD":
                content_type = detect_content_type(fact)
                tags = _extract_tags(fact)
                governance_layer = classify_governance_layer(
                    content_type, protected=False, tags=set(tags),
                )

                surprise = 0.5
                if cfg.enable_surprise:
                    from ..scoring.surprise import compute_surprise, get_centroids
                    centroid_pairs = get_centroids(conn)
                    centroids = [c for _, c in centroid_pairs]
                    surprise = compute_surprise(
                        np.array(embedding, dtype=np.float32), centroids,
                    )

                memory_id = add_memory(
                    conn,
                    content=fact,
                    embedding=embedding,
                    content_type=content_type,
                    source_file=source,
                    author=author,
                    tags=tags,
                    governance_layer=governance_layer,
                    surprise_score=surprise,
                )

                # Auto-link if zettelkasten enabled
                if cfg.enable_zettelkasten:
                    try:
                        from ..graphs.zettelkasten import auto_link
                        auto_link(
                            conn, memory_id, embedding,
                            threshold=cfg.zettel_link_threshold,
                        )
                    except Exception:
                        pass

                stats["added"] += 1
                stats["details"].append({
                    "fact": fact[:100],
                    "action": "ADD",
                    "memory_id": memory_id,
                    "content_type": content_type,
                })

            elif action == "UPDATE":
                target_id = decision.get("memory_id")
                if target_id:
                    update_memory(conn, target_id, content=fact)
                    # Re-embed and update vec table
                    conn.execute("DELETE FROM memory_vec WHERE id = ?", (target_id,))
                    conn.execute(
                        "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                        (target_id, _serialize_f32(embedding)),
                    )
                    conn.commit()
                    stats["updated"] += 1
                    stats["details"].append({
                        "fact": fact[:100],
                        "action": "UPDATE",
                        "memory_id": target_id,
                    })
                else:
                    stats["errors"] += 1

            elif action == "DELETE":
                target_id = decision.get("memory_id")
                if target_id:
                    archive_memory(conn, target_id, reason="superseded")
                    stats["deleted"] += 1
                    stats["details"].append({
                        "fact": fact[:100],
                        "action": "DELETE",
                        "memory_id": target_id,
                    })
                else:
                    stats["errors"] += 1

            else:  # NONE
                stats["skipped"] += 1
                stats["details"].append({
                    "fact": fact[:100],
                    "action": "NONE",
                    "reason": decision.get("reason", "Too similar"),
                })

        except Exception as e:
            stats["errors"] += 1
            stats["details"].append({
                "fact": fact[:100] if fact else "?",
                "action": "ERROR",
                "reason": str(e),
            })

    return stats


def _extract_tags(fact: str) -> list[str]:
    """Extract simple tags from a fact string based on content patterns."""
    tags = []
    fact_lower = fact.lower()

    tag_keywords = {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "rust": "rust",
        "docker": "docker",
        "git": "git",
        "api": "api",
        "database": "database",
        "sqlite": "sqlite",
        "ollama": "ollama",
        "anthropic": "anthropic",
        "claude": "claude",
        "memory": "memory",
        "preference": "preference",
        "identity": "identity",
        "decision": "decision",
        "correction": "correction",
        "project": "project",
    }

    for keyword, tag in tag_keywords.items():
        if keyword in fact_lower:
            tags.append(tag)

    # Detect content type tag
    ctype = detect_content_type(fact)
    if ctype not in tags:
        tags.append(ctype)

    return tags
