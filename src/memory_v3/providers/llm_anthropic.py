"""Anthropic LLM provider."""
from __future__ import annotations

import os

try:
    import anthropic
except ImportError as e:
    raise ImportError(
        "anthropic package required. Install with: pip install 'memory-v3[anthropic]'"
    ) from e


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MEMORY_V3_LLM_MODEL", "claude-haiku-4-5-20251001")
        self._client = anthropic.Anthropic()

    def chat(self, prompt: str, max_tokens: int = 2048) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception:
            return ""
