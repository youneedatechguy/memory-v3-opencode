"""Auto-extraction for memory-v3.

This module collects small high-signal snippets at session boundaries (e.g. git hooks)
and feeds them into the existing 2-pass extraction pipeline.
"""

from __future__ import annotations
