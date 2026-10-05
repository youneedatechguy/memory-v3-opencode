"""
memory-v3 -- Security Module
Credential scanning, integrity manifests, memory sanitization.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .source_policy import build_source_manifest, classify_source_path

# ---------- Credential Scanning ----------

CREDENTIAL_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|secret|token|password|credential|auth)\s*[:=]\s*\S{8,}'),
    re.compile(r'[A-Za-z0-9+/]{40,}={0,2}'),                # Base64 strings (40+ chars)
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),                       # OpenAI keys
    re.compile(r'sk-ant-[a-zA-Z0-9-]{20,}'),                  # Anthropic keys
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),                        # GitHub PATs
    re.compile(r'gho_[a-zA-Z0-9]{36}'),                        # GitHub OAuth tokens
    re.compile(r'(?i)bearer\s+[a-zA-Z0-9._\-]{20,}'),        # Bearer tokens
    re.compile(r'xoxb-[0-9]{10,}-[a-zA-Z0-9]+'),             # Slack bot tokens
    re.compile(r'xoxp-[0-9]{10,}-[a-zA-Z0-9]+'),             # Slack user tokens
    re.compile(r'AKIA[0-9A-Z]{16}'),                           # AWS access keys
    re.compile(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----'),   # Private keys
    re.compile(r'(?i)mongodb(\+srv)?://[^\s]+'),               # MongoDB URIs
    re.compile(r'(?i)postgres(ql)?://[^\s]+'),                 # PostgreSQL URIs
]

# Allowlist: patterns that look like credentials but aren't
ALLOWLIST_PATTERNS = [
    re.compile(r'sk-[a-zA-Z0-9]{3}\.{3}'),                    # Redacted keys (sk-abc...)
    re.compile(r'(?i)example|placeholder|your[_-]?key|xxx'),   # Example values
    re.compile(r'sha256:[a-f0-9]+'),                           # Our own content hashes
]


def scan_for_credentials(text: str) -> list[dict]:
    """
    Scan text for potential credentials.
    Returns list of matches. Empty list = clean.
    """
    matches = []
    for pattern in CREDENTIAL_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group()
            # Check allowlist
            allowed = any(ap.search(matched_text) for ap in ALLOWLIST_PATTERNS)
            if not allowed:
                matches.append({
                    "pattern": pattern.pattern[:50],
                    "match": matched_text[:20] + "...",
                    "position": match.start(),
                })
    return matches


# ---------- Memory Sanitization ----------

INJECTION_PATTERNS = [
    re.compile(r'(?i)ignore\s+(all\s+)?previous\s+instructions'),
    re.compile(r'(?i)you\s+are\s+now\s+a'),
    re.compile(r'(?i)system\s*:\s*you'),
    re.compile(r'(?i)forget\s+(everything|all|your)\s+(you|instructions)'),
    re.compile(r'(?i)new\s+instructions?\s*:'),
    re.compile(r'(?i)override\s+(previous|system|all)'),
]


def sanitize_external_content(text: str) -> tuple[str, list[str]]:
    """
    Sanitize content from external sources (web, documents, emails).
    Returns (cleaned_text, warnings).
    """
    warnings = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            warnings.append(f"Potential injection pattern: {pattern.pattern[:40]}")
    return text, warnings  # We warn but don't strip -- let the caller decide


# ---------- Integrity Manifests ----------

def generate_manifest(vault_path: str | Path) -> dict:
    """Generate SHA-256 manifest for all vault files."""
    vault_path = Path(vault_path)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    for source in build_source_manifest(vault_path):
        rel = source.relative_path
        f = vault_path / rel
        content = f.read_bytes()
        manifest["files"][rel] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
            "last_modified": datetime.fromtimestamp(
                f.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }

    return manifest


def save_manifest(vault_path: str | Path):
    """Generate and save manifest.json to vault root."""
    vault_path = Path(vault_path)
    manifest = generate_manifest(vault_path)
    manifest_file = vault_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    return manifest


def check_integrity(vault_path: str | Path) -> dict:
    """
    Check vault files against saved manifest.
    Returns dict with 'ok', 'modified', 'new', 'missing' lists.
    """
    vault_path = Path(vault_path)
    manifest_file = vault_path / "manifest.json"

    if not manifest_file.exists():
        return {"error": "No manifest.json found. Run save_manifest() first."}

    manifest = json.loads(manifest_file.read_text())
    current = generate_manifest(vault_path)

    result = {"ok": [], "modified": [], "new": [], "missing": [], "excluded_by_policy": []}

    old_files = set(manifest["files"].keys())
    new_files = set(current["files"].keys())

    # Check existing files
    for f in old_files & new_files:
        if manifest["files"][f]["sha256"] == current["files"][f]["sha256"]:
            result["ok"].append(f)
        else:
            result["modified"].append(f)

    # New files
    for f in new_files - old_files:
        if f != "manifest.json":
            result["new"].append(f)

    # Missing files
    for f in old_files - new_files:
        path = vault_path / f
        if path.exists() and classify_source_path(path, vault_path) is None:
            result["excluded_by_policy"].append(f)
        else:
            result["missing"].append(f)

    return result
