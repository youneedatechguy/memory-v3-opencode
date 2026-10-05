"""OpenAI embedding provider."""
from __future__ import annotations

import os
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace

from memory_v3.config import get_config

openai = SimpleNamespace(OpenAI=None)


class OpenAIEmbeddingError(RuntimeError):
    pass


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _is_retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return status == 429 or 500 <= status <= 599
    exception_names = {cls.__name__ for cls in type(exc).__mro__}
    if exception_names & {"APIConnectionError", "APITimeoutError"}:
        return True
    error_type = getattr(exc, "type", None)
    if error_type in {"api_connection_error", "timeout"}:
        return True
    return False


def _headers(exc: Exception):
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    return headers or {}


def _header_get(headers, key: str) -> str | None:
    getter = getattr(headers, "get", None)
    if getter is not None:
        value = getter(key)
        if value is None:
            value = getter(key.lower())
        return value
    return None


def _parse_retry_after(value: str | None, now: float) -> float | None:
    if value is None:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        delay = date.timestamp() - now
    if delay < 0:
        return None
    return delay


class OpenAIEmbedder:
    def __init__(self, model: str | None = None, dim: int | None = None):
        self.model = model or os.environ.get("MEMORY_V3_EMBED_MODEL", "text-embedding-3-small")
        self.dim = dim or int(os.environ.get("MEMORY_V3_EMBED_DIM", "1536"))
        self._client = None
        self._sleep = time.sleep
        self._monotonic = time.monotonic
        self._time = time.time

    def _get_client(self):
        if self._client is None:
            global openai
            if openai.OpenAI is None:
                try:
                    import openai as openai_module
                except ImportError as e:
                    raise ImportError(
                        "openai package required for OpenAI embeddings. "
                        "Install with: pip install 'memory-v3[openai]'"
                    ) from e
                openai = openai_module

            self._client = openai.OpenAI(max_retries=0)
        return self._client

    def _request_with_retry(self, input_text):
        cfg = get_config()
        started = self._monotonic()
        last_exc = None
        for attempt in range(1, cfg.embed_retry_max_attempts + 1):
            try:
                return self._get_client().embeddings.create(model=self.model, input=input_text)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                if attempt >= cfg.embed_retry_max_attempts:
                    break
                delay = self._delay_for_attempt(exc, attempt, started, cfg)
                if delay is None:
                    break
                self._sleep(delay)
        raise OpenAIEmbeddingError("OpenAI embedding request failed after retries") from last_exc

    def _delay_for_attempt(self, exc: Exception, attempt: int, started: float, cfg) -> float | None:
        remaining = cfg.embed_retry_max_total_seconds - (self._monotonic() - started)
        if remaining < 0:
            return None
        retry_after = _parse_retry_after(
            _header_get(_headers(exc), "Retry-After"),
            self._time(),
        )
        if (
            retry_after is not None
            and retry_after <= cfg.embed_retry_max_seconds
            and retry_after <= remaining
        ):
            return retry_after
        delay = min(
            cfg.embed_retry_base_seconds * (2 ** (attempt - 1)),
            cfg.embed_retry_max_seconds,
        )
        if delay > remaining:
            delay = remaining
        return delay

    def embed(self, text: str) -> list[float]:
        resp = self._request_with_retry(text)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._request_with_retry(texts)
        return [item.embedding for item in resp.data]
