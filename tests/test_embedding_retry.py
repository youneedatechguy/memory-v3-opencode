import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_v3.config import get_config, reset_config
from memory_v3.providers.embedder_openai import OpenAIEmbedder, OpenAIEmbeddingError

TEST_EMBED_MODEL = "test-embed-model"
TEST_EMBED_PROVIDER = "test-provider"
TEST_EMBED_DIM = 768


class FakeResponse:
    def __init__(self, embeddings):
        self.data = [type("Item", (), {"embedding": embedding}) for embedding in embeddings]


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None):
        super().__init__(f"http {status_code}")
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after


class FakeSDKResponse:
    def __init__(self, status_code: int, retry_after: str | None = None):
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["retry-after"] = retry_after


class FakeAPIStatusError(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None):
        super().__init__(f"sdk status {status_code}")
        self.response = FakeSDKResponse(status_code, retry_after)


class RateLimitError(FakeAPIStatusError):
    pass


class APIStatusError(FakeAPIStatusError):
    pass


class APIConnectionError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class FakeEmbeddings:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, input))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.embeddings = FakeEmbeddings(outcomes)


class AdvancingClock:
    def __init__(self):
        self.now = 100.0
        self.wall = 1_700_000_000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def time(self):
        return self.wall

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        self.wall += seconds


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    for key in [
        "MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS",
        "MEMORY_V3_EMBED_RETRY_BASE_SECONDS",
        "MEMORY_V3_EMBED_RETRY_MAX_SECONDS",
        "MEMORY_V3_EMBED_RETRY_MAX_TOTAL_SECONDS",
    ]:
        monkeypatch.delenv(key, raising=False)
    reset_config()
    yield
    reset_config()


def embedder(outcomes, clock: AdvancingClock | None = None) -> OpenAIEmbedder:
    instance = OpenAIEmbedder(model="fake-model", dim=3)
    instance._client = FakeClient(outcomes)
    if clock is not None:
        instance._sleep = clock.sleep
        instance._monotonic = clock.monotonic
        instance._time = clock.time
    return instance


def test_openai_client_disables_sdk_retries(monkeypatch):
    import memory_v3.providers.embedder_openai as embedder_openai

    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.embeddings = FakeEmbeddings([FakeResponse([[1.0, 2.0, 3.0]])])

    monkeypatch.setattr(embedder_openai, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    provider = OpenAIEmbedder(model="fake-model", dim=3)

    assert provider.embed("hello") == [1.0, 2.0, 3.0]
    assert calls == [{"max_retries": 0}]


def test_retry_config_defaults_and_exact_envs(monkeypatch):
    cfg = get_config()

    assert cfg.embed_retry_max_attempts == 3
    assert cfg.embed_retry_base_seconds == 0.5
    assert cfg.embed_retry_max_seconds == 5.0
    assert cfg.embed_retry_max_total_seconds == 10.0

    reset_config()
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "0.25")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_SECONDS", "2")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_TOTAL_SECONDS", "3.5")

    cfg = get_config()

    assert cfg.embed_retry_max_attempts == 4
    assert cfg.embed_retry_base_seconds == 0.25
    assert cfg.embed_retry_max_seconds == 2.0
    assert cfg.embed_retry_max_total_seconds == 3.5


@pytest.mark.parametrize(
    ("env", "value", "match"),
    [
        ("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "0", "positive integer"),
        ("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "-0.1", "greater than or equal to 0"),
        ("MEMORY_V3_EMBED_RETRY_MAX_SECONDS", "-1", "greater than or equal to 0"),
        ("MEMORY_V3_EMBED_RETRY_MAX_TOTAL_SECONDS", "-1", "greater than or equal to 0"),
    ],
)
def test_retry_config_validation(monkeypatch, env, value, match):
    monkeypatch.setenv(env, value)
    reset_config()

    with pytest.raises(ValueError, match=match):
        get_config()


def test_retry_config_rejects_max_sleep_below_base(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "2")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_SECONDS", "1")
    reset_config()

    with pytest.raises(ValueError, match="MAX_SECONDS"):
        get_config()


def test_429_then_success(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "0.25")
    reset_config()
    clock = AdvancingClock()
    provider = embedder([FakeHTTPError(429), FakeResponse([[1.0, 2.0, 3.0]])], clock)

    assert provider.embed("hello") == [1.0, 2.0, 3.0]
    assert clock.sleeps == [0.25]
    assert provider._client.embeddings.calls == [("fake-model", "hello"), ("fake-model", "hello")]


def test_repeated_5xx_then_success(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "0.5")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "3")
    reset_config()
    clock = AdvancingClock()
    provider = embedder(
        [FakeHTTPError(500), FakeHTTPError(503), FakeResponse([[4.0, 5.0, 6.0]])],
        clock,
    )

    assert provider.embed("hello") == [4.0, 5.0, 6.0]
    assert clock.sleeps == [0.5, 1.0]
    assert len(provider._client.embeddings.calls) == 3


def test_nonretryable_400_and_arbitrary_error_fail_immediately():
    for error in [FakeHTTPError(400), RuntimeError("boom")]:
        clock = AdvancingClock()
        provider = embedder([error], clock)

        with pytest.raises(type(error)):
            provider.embed("hello")

        assert clock.sleeps == []
        assert len(provider._client.embeddings.calls) == 1


def test_sdk_status_shapes_retry_429_and_5xx_but_not_4xx(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "2")
    reset_config()
    rate_limited = embedder(
        [RateLimitError(429), FakeResponse([[1.0, 0.0, 0.0]])],
        AdvancingClock(),
    )
    server_error = embedder(
        [APIStatusError(503), FakeResponse([[0.0, 1.0, 0.0]])],
        AdvancingClock(),
    )
    client_error = embedder([APIStatusError(400)], AdvancingClock())

    assert rate_limited.embed("rate") == [1.0, 0.0, 0.0]
    assert server_error.embed("server") == [0.0, 1.0, 0.0]
    with pytest.raises(APIStatusError):
        client_error.embed("client")
    assert len(client_error._client.embeddings.calls) == 1


@pytest.mark.parametrize("error", [APIConnectionError("connection"), APITimeoutError("timeout")])
def test_sdk_connection_and_timeout_errors_retry_without_status(monkeypatch, error):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "2")
    reset_config()
    clock = AdvancingClock()
    provider = embedder([error, FakeResponse([[3.0, 2.0, 1.0]])], clock)

    assert provider.embed("transient") == [3.0, 2.0, 1.0]
    assert clock.sleeps == [0.5]
    assert len(provider._client.embeddings.calls) == 2


def test_max_attempts_and_exact_delay_sequence_with_caps(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "1")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_SECONDS", "2.5")
    reset_config()
    clock = AdvancingClock()
    provider = embedder(
        [FakeHTTPError(500), FakeHTTPError(500), FakeHTTPError(500), FakeHTTPError(500)],
        clock,
    )

    with pytest.raises(OpenAIEmbeddingError):
        provider.embed("hello")

    assert clock.sleeps == [1.0, 2.0, 2.5]
    assert len(provider._client.embeddings.calls) == 4


def test_max_total_sleep_caps_final_delay(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "2")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_SECONDS", "10")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_TOTAL_SECONDS", "3")
    reset_config()
    clock = AdvancingClock()
    provider = embedder(
        [
            FakeHTTPError(500),
            FakeHTTPError(500),
            FakeHTTPError(500),
            FakeResponse([[1.0, 1.0, 1.0]]),
        ],
        clock,
    )

    assert provider.embed("hello") == [1.0, 1.0, 1.0]
    assert clock.sleeps == [2.0, 1.0, 0.0]


def test_retry_after_honored_only_when_parsed_and_within_caps(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "1")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_SECONDS", "5")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_TOTAL_SECONDS", "6")
    reset_config()
    clock = AdvancingClock()
    provider = embedder(
        [
            FakeHTTPError(429, "3"),
            FakeHTTPError(429, "99"),
            FakeHTTPError(429, "not-a-delay"),
            FakeResponse([[2.0, 2.0, 2.0]]),
        ],
        clock,
    )

    assert provider.embed("hello") == [2.0, 2.0, 2.0]
    assert clock.sleeps == [3.0, 2.0, 1.0]


def test_retry_after_http_date_uses_wall_clock_with_monotonic_elapsed_caps(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_BASE_SECONDS", "1")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_SECONDS", "10")
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_TOTAL_SECONDS", "5")
    reset_config()
    clock = AdvancingClock()
    clock.wall = 0.0
    provider = embedder(
        [
            FakeHTTPError(429, "Thu, 01 Jan 1970 00:00:03 GMT"),
            FakeHTTPError(429, "Thu, 01 Jan 1970 00:00:09 GMT"),
            FakeResponse([[7.0, 8.0, 9.0]]),
        ],
        clock,
    )

    assert provider.embed("dated") == [7.0, 8.0, 9.0]
    assert clock.sleeps == [3.0, 2.0]


def test_single_and_batch_calls_have_one_retry_loop_each(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_EMBED_RETRY_MAX_ATTEMPTS", "2")
    reset_config()
    single = embedder([FakeHTTPError(429), FakeResponse([[1.0, 0.0, 0.0]])], AdvancingClock())
    batch = embedder(
        [FakeHTTPError(500), FakeResponse([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])],
        AdvancingClock(),
    )

    assert single.embed("one") == [1.0, 0.0, 0.0]
    assert batch.embed_batch(["one", "two"]) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert len(single._client.embeddings.calls) == 2
    assert len(batch._client.embeddings.calls) == 2
    assert batch._client.embeddings.calls == [
        ("fake-model", ["one", "two"]),
        ("fake-model", ["one", "two"]),
    ]


def test_cache_remains_empty_on_embedding_failure(monkeypatch, tmp_path):
    import memory_v3.embeddings as embeddings

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(embeddings, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        embeddings,
        "embed_text",
        lambda text: (_ for _ in ()).throw(
            OpenAIEmbeddingError("OpenAI embedding request failed after retries")
        ),
    )

    with pytest.raises(OpenAIEmbeddingError):
        embeddings.embed_with_cache("uncached")

    assert list(cache_dir.glob("*.json")) == []


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def long_note(marker: str) -> str:
    lines = ["# Long\n"] + [f"line {index:03d} {'x' * 90}\n" for index in range(90)]
    lines[75] = marker + "\n"
    return "".join(lines)


class HashEmbedder:
    def __init__(self, failures_after: int | None = None):
        self.inputs = []
        self.failures_after = failures_after

    def embed(self, text: str) -> list[float]:
        if self.failures_after is not None and len(self.inputs) >= self.failures_after:
            raise OpenAIEmbeddingError("OpenAI embedding request failed after retries")
        self.inputs.append(text)
        digest = hashlib.sha256(text.encode()).digest()
        return [(digest[i % len(digest)] / 255.0) for i in range(TEST_EMBED_DIM)]


def configure_reindex(monkeypatch, tmp_path, vault: Path) -> None:
    import memory_v3.db as memory_db

    monkeypatch.setenv("MEMORY_V3_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("MEMORY_V3_VAULT", str(vault))
    monkeypatch.setenv("MEMORY_V3_GRAPH_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("MEMORY_V3_EMBED_MODEL", TEST_EMBED_MODEL)
    monkeypatch.setenv("MEMORY_V3_EMBED_PROVIDER", TEST_EMBED_PROVIDER)
    monkeypatch.setenv("MEMORY_V3_EMBED_DIM", str(TEST_EMBED_DIM))
    monkeypatch.setenv("MEMORY_V3_INDEXING_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_OWNER", "owner-a")
    # db.EMBEDDING_DIM is read from the ambient env at import time and fixes the
    # memory_vec dimension, so pin it to the same dim this test configures.
    monkeypatch.setattr(memory_db, "EMBEDDING_DIM", TEST_EMBED_DIM)
    reset_config()


def source_state(conn, path: str, *, keyword: str) -> dict:
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT id, content, source_hash, source_id, chunk_id, chunk_index, chunk_count
            FROM memories
            WHERE source_file = ? AND owner = 'owner-a' AND record_kind = 'derived'
            ORDER BY chunk_index, id
            """,
            (path,),
        ).fetchall()
    ]
    vector_rows = [
        {"id": row["id"], "embedding": bytes(row["embedding"])}
        for row in conn.execute("SELECT id, embedding FROM memory_vec ORDER BY id").fetchall()
    ]
    file_hash = conn.execute(
        "SELECT content_hash, chunk_count FROM file_hashes WHERE file_path = ?",
        (path,),
    ).fetchone()
    return {
        "memory_rows": rows,
        "memory_count": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
        "vector_rows": vector_rows,
        "vector_count": conn.execute("SELECT COUNT(*) FROM memory_vec").fetchone()[0],
        "fts_count": conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0],
        "fts_results": [
            dict(row)
            for row in conn.execute(
                """
                SELECT m.id, m.content, m.chunk_id, m.chunk_index, m.chunk_count
                FROM memory_fts f
                JOIN memories m ON m.id = f.rowid
                WHERE memory_fts MATCH ?
                ORDER BY m.chunk_index, m.id
                """,
                (keyword,),
            ).fetchall()
        ],
        "file_hash": None if file_hash is None else dict(file_hash),
    }


def test_reindex_retry_exhaustion_preserves_prior_source_then_replaces_atomically(
    monkeypatch, tmp_path
):
    import memory_v3.server as server

    if server._conn is not None:
        server._conn.close()
    server._conn = None
    vault = tmp_path / "vault"
    path = vault / "research/long.md"
    write(path, long_note("oldmarker"))
    configure_reindex(monkeypatch, tmp_path, vault)
    monkeypatch.setattr(server, "get_embedder", lambda: HashEmbedder())
    assert json.loads(server.reindex())["errors"] == 0
    before = source_state(server._conn, "research/long.md", keyword="oldmarker")
    old_chunk_ids = {row["chunk_id"] for row in before["memory_rows"]}

    write(path, long_note("newmarker"))
    failing = HashEmbedder(failures_after=1)
    monkeypatch.setattr(server, "get_embedder", lambda: failing)
    failed = json.loads(server.reindex())
    after = source_state(server._conn, "research/long.md", keyword="oldmarker")

    assert failed["errors"] == 1
    assert failed["source_errors"][0]["exception_class"] == "OpenAIEmbeddingError"
    assert failed["source_errors"][0]["source_rows_unchanged"] is True
    assert failed["source_errors"][0]["file_hash_unchanged"] is True
    assert after == before
    assert source_state(server._conn, "research/long.md", keyword="newmarker")["fts_results"] == []

    succeeding = HashEmbedder()
    monkeypatch.setattr(server, "get_embedder", lambda: succeeding)
    result = json.loads(server.reindex())
    final = source_state(server._conn, "research/long.md", keyword="newmarker")

    assert result["errors"] == 0
    assert final["file_hash"]["content_hash"] != before["file_hash"]["content_hash"]
    assert final["memory_count"] == before["memory_count"]
    assert final["vector_count"] == before["vector_count"]
    assert final["fts_count"] == before["fts_count"]
    assert {row["chunk_id"] for row in final["memory_rows"]}.isdisjoint(old_chunk_ids)
    assert {row["source_hash"] for row in final["memory_rows"]} == {
        final["file_hash"]["content_hash"]
    }
    assert all("oldmarker" not in row["content"] for row in final["memory_rows"])
    assert any("newmarker" in row["content"] for row in final["memory_rows"])
    assert final["fts_results"]
    assert source_state(server._conn, "research/long.md", keyword="oldmarker")["fts_results"] == []
    assert len(succeeding.inputs) > 0
