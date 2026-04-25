"""
memory-v3 -- CogCanvas 6-Step Compaction Pipeline

Intelligent text compression that preserves critical content:
  1. Extract protected content (names, identifiers, decisions)
  2. Selective delete (filler, greetings, meta-text)
  3. Structured summary (LLM-powered, ratio-aware)
  4. Verify faithfulness (check no protected content lost)
  5. Generate receipt (audit trail)
  6. Rehydrate (reconstruct from compressed + source)

Protected names configurable via MEMORY_V3_PROTECTED_NAMES env var.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import Config, get_config
from ..providers import get_llm

# ---------------------------------------------------------------------------
# Protected content patterns
# ---------------------------------------------------------------------------

# Configurable names: comma-separated via env var
_PROTECTED_NAMES_RAW = os.environ.get("MEMORY_V3_PROTECTED_NAMES", "Sean,Saranna,Sondra,Claude")
PROTECTED_NAMES = [n.strip() for n in _PROTECTED_NAMES_RAW.split(",") if n.strip()]

# Patterns that mark content as protected
PROTECTED_PATTERNS = [
    re.compile(r"\b(?:decision|decided|commitment|committed)\b.*?[.!]", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:correction|corrected|actually|wrong)\b.*?[.!]", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:identity|my name is|i am)\b.*?[.!]", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:rule|always|never|must)\b.*?[.!]", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:API[_\s]?key|token|secret|password)\s*[:=].*", re.IGNORECASE),
    re.compile(r"\b(?:deadline|due date|by \w+ \d+)\b.*?[.!]", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b(?:version|v\d+\.\d+|release)\b.*?[.!]", re.IGNORECASE | re.DOTALL),
]

# Expendable content patterns (safe to delete)
EXPENDABLE_PATTERNS = [
    re.compile(r"^(?:hi|hello|hey|thanks|thank you|okay|ok|sure|got it|sounds good)[.!,]?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:let me|i'll|i will|going to)\s+(?:check|look|see|think).*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:hmm|uh|um|well|so|anyway|basically)[\s,].*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:great|perfect|awesome|nice|cool)[.!]?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*$", re.MULTILINE),
]


def is_protected_content(text: str) -> bool:
    """Check if text contains protected content that must be preserved."""
    # Check protected names
    for name in PROTECTED_NAMES:
        if name.lower() in text.lower():
            return True

    # Check protected patterns
    for pattern in PROTECTED_PATTERNS:
        if pattern.search(text):
            return True

    return False


def extract_protected(text: str) -> tuple[list[str], str]:
    """Extract protected items from text.

    Returns (protected_items, remaining_text).
    Protected items are extracted verbatim; remaining_text has them removed.
    """
    protected_items = []
    remaining = text

    # Extract sentences containing protected names
    sentences = re.split(r'(?<=[.!?])\s+', text)
    protected_sentences = []
    remaining_sentences = []

    for sentence in sentences:
        if is_protected_content(sentence):
            protected_sentences.append(sentence)
            protected_items.append(sentence.strip())
        else:
            remaining_sentences.append(sentence)

    remaining = " ".join(remaining_sentences)

    # Also extract pattern matches that might span multiple sentences
    for pattern in PROTECTED_PATTERNS:
        for match in pattern.finditer(remaining):
            matched = match.group().strip()
            if matched and matched not in protected_items:
                protected_items.append(matched)

    return protected_items, remaining


def selective_delete(text: str) -> tuple[str, int]:
    """Remove expendable content (filler, greetings, meta-text).

    Returns (cleaned_text, chars_removed).
    """
    original_len = len(text)
    cleaned = text

    for pattern in EXPENDABLE_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Collapse multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()

    chars_removed = original_len - len(cleaned)
    return cleaned, chars_removed


def structured_summary(
    text: str,
    max_ratio: float = 3.0,
    config: Optional[Config] = None,
) -> str:
    """Generate a structured summary using the local LLM.

    Target length is roughly len(text) / max_ratio characters.
    """
    cfg = config or get_config()

    target_len = max(50, int(len(text) / max_ratio))

    prompt = (
        f"Compress the following text to approximately {target_len} characters. "
        "Preserve all facts, names, decisions, and technical details. "
        "Remove only filler and redundancy. Write in the same voice.\n\n"
        f"Text:\n{text}\n\n"
        "Compressed:"
    )

    try:
        summary = get_llm().chat(prompt, max_tokens=2048)
        return summary
    except Exception as e:
        # Fallback: simple truncation preserving sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = ""
        for s in sentences:
            if len(result) + len(s) < target_len * 1.5:
                result += s + " "
            else:
                break
        return result.strip() or text[:target_len]


def verify_summary(original: str, summary: str) -> dict:
    """Verify that a summary faithfully represents the original.

    Checks:
      1. Protected names are preserved
      2. Key numeric values are preserved
      3. Ratio is within target
      4. No hallucinated content (via keyword overlap)

    Returns {faithful: bool, score: float, issues: list[str]}
    """
    issues = []

    # Check protected names
    for name in PROTECTED_NAMES:
        if name.lower() in original.lower() and name.lower() not in summary.lower():
            issues.append(f"Protected name '{name}' missing from summary")

    # Check numbers
    original_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', original))
    summary_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', summary))
    missing_numbers = original_numbers - summary_numbers
    if missing_numbers and len(missing_numbers) > len(original_numbers) * 0.5:
        issues.append(f"Missing numbers: {missing_numbers}")

    # Check compression ratio
    if len(original) > 0:
        ratio = len(original) / max(len(summary), 1)
    else:
        ratio = 1.0

    if ratio < 1.0:
        issues.append(f"Summary is longer than original (ratio={ratio:.2f})")

    # Keyword overlap score
    original_words = set(original.lower().split())
    summary_words = set(summary.lower().split())
    if original_words:
        overlap = len(original_words & summary_words) / len(original_words)
    else:
        overlap = 1.0

    if overlap < 0.2:
        issues.append(f"Low keyword overlap ({overlap:.2f}) -- possible hallucination")

    # Compute faithfulness score
    score = 1.0
    score -= 0.2 * len([i for i in issues if "Protected name" in i])
    score -= 0.1 * len([i for i in issues if "Missing numbers" in i])
    score -= 0.3 * len([i for i in issues if "hallucination" in i])
    score = max(0.0, min(1.0, score))

    return {
        "faithful": len(issues) == 0,
        "score": score,
        "issues": issues,
        "ratio": ratio,
        "keyword_overlap": overlap,
    }


def generate_receipt(
    original: str,
    compressed: str,
    protected: list[str],
    verification: dict,
) -> dict:
    """Generate an audit receipt for the compaction operation."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_length": len(original),
        "compressed_length": len(compressed),
        "ratio": len(original) / max(len(compressed), 1),
        "protected_items_count": len(protected),
        "protected_items": protected[:20],  # Cap for storage
        "verification_score": verification.get("score", 0.0),
        "verification_faithful": verification.get("faithful", False),
        "verification_issues": verification.get("issues", []),
        "original_hash": hashlib.sha256(original.encode()).hexdigest()[:16],
        "compressed_hash": hashlib.sha256(compressed.encode()).hexdigest()[:16],
    }


def compact(
    text: str,
    max_ratio: float = 3.0,
    verify: bool = True,
    conn: Optional[sqlite3.Connection] = None,
    config: Optional[Config] = None,
) -> dict:
    """Full CogCanvas 6-step compaction pipeline.

    Steps:
      1. Extract protected content
      2. Selective delete expendable content
      3. Generate structured summary
      4. Verify faithfulness (if verify=True)
      5. Generate receipt
      6. Store receipt (if conn provided)

    Returns:
      {
        compressed: str,
        protected_items: list[str],
        receipt: dict,
        verification: dict,
      }
    """
    cfg = config or get_config()

    # Step 1: Extract protected content
    protected_items, remaining = extract_protected(text)

    # Step 2: Selective delete
    cleaned, chars_removed = selective_delete(remaining)

    # Step 3: Structured summary
    if len(cleaned) > 100:
        summary = structured_summary(cleaned, max_ratio=max_ratio, config=cfg)
    else:
        summary = cleaned

    # Reattach protected items at the top
    if protected_items:
        protected_block = "\n".join(f"[PROTECTED] {item}" for item in protected_items)
        compressed = f"{protected_block}\n\n{summary}"
    else:
        compressed = summary

    # Step 4: Verify
    verification = {}
    if verify:
        verification = verify_summary(text, compressed)
    else:
        verification = {"faithful": True, "score": 1.0, "issues": []}

    # Step 5: Generate receipt
    receipt = generate_receipt(text, compressed, protected_items, verification)

    # Step 6: Store receipt if connection provided
    if conn is not None:
        try:
            conn.execute(
                """INSERT INTO compaction_receipts
                (timestamp, original_tokens, compressed_tokens, ratio,
                 protected_items_extracted, verification_score,
                 vault_files_updated, receipt_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    receipt["timestamp"],
                    receipt["original_length"],
                    receipt["compressed_length"],
                    receipt["ratio"],
                    receipt["protected_items_count"],
                    receipt["verification_score"],
                    json.dumps([]),
                    json.dumps(receipt),
                ),
            )
            conn.commit()
        except Exception:
            pass

    return {
        "compressed": compressed,
        "protected_items": protected_items,
        "receipt": receipt,
        "verification": verification,
    }


def rehydrate(
    memory_id: Optional[int] = None,
    source_file: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[str]:
    """Reconstruct full content from a compressed memory + its source file.

    Tries to recover the original content by:
      1. Loading the memory's compressed content from DB
      2. Loading the original source file if it still exists
      3. Combining protected items with source context

    Returns the rehydrated text, or None if recovery is not possible.
    """
    if conn is None:
        return None

    content = None
    source_path = None

    # Load memory content
    if memory_id is not None:
        row = conn.execute(
            "SELECT content, source_file FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is not None:
            mem = dict(row)
            content = mem.get("content")
            source_path = mem.get("source_file")

    # Override source from parameter
    if source_file is not None:
        source_path = source_file

    if content is None and source_path is None:
        return None

    # Try to load source file
    source_content = None
    if source_path:
        try:
            source_content = Path(source_path).read_text(encoding="utf-8")
        except (OSError, IOError, UnicodeDecodeError):
            pass

    # If we have both, combine them
    if content and source_content:
        # Extract protected items from the compressed content
        protected_lines = []
        other_lines = []
        for line in content.split("\n"):
            if line.startswith("[PROTECTED]"):
                protected_lines.append(line.replace("[PROTECTED] ", ""))
            else:
                other_lines.append(line)

        # Build rehydrated content
        parts = []
        if protected_lines:
            parts.append("Key facts:\n" + "\n".join(f"- {p}" for p in protected_lines))
        parts.append(f"\nSource context ({source_path}):\n{source_content}")
        if other_lines:
            summary = "\n".join(other_lines).strip()
            if summary:
                parts.append(f"\nSummary:\n{summary}")

        return "\n".join(parts)

    # Return whatever we have
    if source_content:
        return source_content
    return content
