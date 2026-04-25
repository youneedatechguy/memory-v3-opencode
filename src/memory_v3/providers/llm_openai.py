"""OpenAI LLM provider."""
from __future__ import annotations

import os

try:
    import openai
except ImportError as e:
    raise ImportError(
        "openai package required. Install with: pip install 'memory-v3[openai]'"
    ) from e


class OpenAILLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MEMORY_V3_LLM_MODEL", "gpt-4o-mini")
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = openai.OpenAI()
        return self._client

    def chat(self, prompt: str, max_tokens: int = 2048) -> str:
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""
