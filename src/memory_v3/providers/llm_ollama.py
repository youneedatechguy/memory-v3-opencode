"""Ollama LLM provider."""
from __future__ import annotations

import os

import ollama


class OllamaLLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("MEMORY_V3_LLM_MODEL", "qwen2.5:3b")

    def chat(self, prompt: str, max_tokens: int = 2048) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": max_tokens},
            )
            return response["message"]["content"].strip()
        except Exception:
            return ""
