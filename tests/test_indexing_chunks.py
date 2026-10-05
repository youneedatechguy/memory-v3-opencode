import pytest

from memory_v3.indexing import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_LINES,
    DEFAULT_TARGET_CHARS,
    build_source_chunks,
    make_index_source_id,
    make_source_version_key,
)


def body_lines(chunks):
    return [chunk.body.splitlines() for chunk in chunks]


def non_overlap_lines(chunks):
    seen_until = 0
    lines = []
    for chunk in chunks:
        split = chunk.body.splitlines()
        skip = max(0, seen_until - chunk.line_start + 1)
        lines.extend(split[skip:])
        seen_until = max(seen_until, chunk.line_end)
    return lines


def test_short_note_is_one_chunk_with_stable_source_context():
    text = "# Title\n\nBody line.\n"

    chunks = build_source_chunks("owner-a", "docs", "docs/note.md", text)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_index == 0
    assert chunk.chunk_count == 1
    assert chunk.relative_path == "docs/note.md"
    assert chunk.line_start == 1
    assert chunk.line_end == 3
    assert chunk.heading_path == ("Title",)
    assert chunk.body == text
    assert chunk.source_hash
    assert chunk.source_id.startswith("src:")
    assert chunk.source_version_key.startswith("ver:")
    assert chunk.chunk_id.startswith("chk:")
    assert "Source: docs/note.md" in chunk.embedding_text
    assert "Lines: 1-3" in chunk.embedding_text
    assert "Headings: Title" in chunk.embedding_text
    assert chunk.embedding_text.endswith(text)


def test_long_note_has_multiple_chunks_and_final_quarter_marker():
    lines = ["# Long\n"] + [f"line {index:03d} {'x' * 80}\n" for index in range(80)]
    lines[70] = "FINAL_QUARTER_SEARCHABLE_MARKER\n"
    text = "".join(lines)

    chunks = build_source_chunks(
        "owner-a",
        "research",
        "research/long.md",
        text,
        target_chars=700,
        max_chars=1000,
        overlap_lines=2,
    )

    assert len(chunks) > 1
    assert any("FINAL_QUARTER_SEARCHABLE_MARKER" in chunk.embedding_text for chunk in chunks)
    assert non_overlap_lines(chunks) == text.splitlines()
    assert chunks[-1].line_end == len(text.splitlines())


def test_nested_headings_and_exact_line_spans_are_preserved():
    text = "\n".join(
        [
            "# Root",
            "intro",
            "## Child",
            "child body",
            "### Grandchild",
            "grand body",
            "## Sibling",
            "sibling body",
        ]
    )
    text += "\n"

    chunks = build_source_chunks(
        "owner-a",
        "docs",
        "docs/headings.md",
        text,
        target_chars=20,
        max_chars=120,
        overlap_lines=0,
    )

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [
        (1, 3),
        (4, 5),
        (6, 7),
        (8, 8),
    ]
    assert [chunk.heading_path for chunk in chunks] == [
        ("Root",),
        ("Root", "Child"),
        ("Root", "Child", "Grandchild"),
        ("Root", "Sibling"),
    ]


def test_overlap_is_bounded_and_non_overlap_lines_are_neither_omitted_nor_duplicated():
    text = "".join(f"line-{index}\n" for index in range(1, 21))

    chunks = build_source_chunks(
        "owner-a",
        "docs",
        "docs/overlap.md",
        text,
        target_chars=35,
        max_chars=60,
        overlap_lines=3,
    )

    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:]):
        overlap = previous.line_end - current.line_start + 1
        assert 0 <= overlap <= 3
    assert non_overlap_lines(chunks) == text.splitlines()


def test_hard_max_is_respected_by_splitting_long_source_lines():
    long_line = "L" * 80 + "\n"
    text = "short-1\nshort-2\n" + long_line + "short-3\n" + "short-4\n"

    chunks = build_source_chunks(
        "owner-a",
        "docs",
        "docs/hard-max.md",
        text,
        target_chars=20,
        max_chars=80,
        overlap_lines=2,
    )

    assert all(len(chunk.embedding_text) <= 80 for chunk in chunks)
    assert "".join(non_overlap_lines(chunks)) == text.replace("\n", "")


def test_embedding_text_hard_max_includes_long_path_and_deep_headings():
    text = "# Root\n## Child\n### Grandchild\n" + "".join(
        f"body-{index:02d}\n" for index in range(1, 11)
    )
    chunks = build_source_chunks(
        "owner-a",
        "docs",
        "docs/very/long/path/that/counts/against/metadata/note.md",
        text,
        target_chars=120,
        max_chars=145,
        overlap_lines=2,
    )

    assert len(chunks) > 1
    assert all(len(chunk.embedding_text) <= 145 for chunk in chunks)
    assert any(chunk.heading_path == ("Root", "Child", "Grandchild") for chunk in chunks)
    assert non_overlap_lines(chunks) == text.splitlines()


def test_single_long_line_is_split_within_hard_max():
    text = "# Heading\n" + "x" * 80 + "\n"
    chunks = build_source_chunks(
        "owner-a",
        "docs",
        "docs/tiny.md",
        text,
        target_chars=10,
        max_chars=70,
        overlap_lines=1,
    )

    assert all(len(chunk.embedding_text) <= 70 for chunk in chunks)
    assert "".join(non_overlap_lines(chunks)) == text.replace("\n", "")


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/note.md",
        "C:/Users/base/note.md",
        "C:\\Users\\base\\note.md",
        "C:note.md",
        "//server/share/note.md",
        "\\\\server\\share\\note.md",
        "",
        ".",
        "./",
        "/",
        "../note.md",
        "docs/../note.md",
        "docs/sub/../../note.md",
    ],
)
def test_canonical_source_path_rejects_unsafe_paths(path):
    with pytest.raises(ValueError):
        build_source_chunks("owner-a", "docs", path, "body\n")


def test_canonical_source_path_normalizes_safe_separators_and_dot_segments():
    chunks = build_source_chunks("owner-a", "docs", "docs\\./safe\\note.md", "body\n")

    assert chunks[0].relative_path == "docs/safe/note.md"


def test_repeated_run_produces_identical_dataclasses_and_ids():
    text = "# Stable\n\n" + "".join(f"line {index}\n" for index in range(20))

    first = build_source_chunks("owner-a", "docs", "docs/stable.md", text, target_chars=50)
    second = build_source_chunks("owner-a", "docs", "docs/stable.md", text, target_chars=50)

    assert first == second


def test_empty_content_returns_no_chunks():
    assert build_source_chunks("owner-a", "docs", "docs/empty.md", "") == ()


def test_changed_content_keeps_source_identity_but_gets_new_version_and_chunk_ids():
    old = "# Note\n\nold body\n"
    new = "# Note\n\nnew body\n"

    old_chunks = build_source_chunks("owner-a", "docs", "docs/change.md", old)
    new_chunks = build_source_chunks("owner-a", "docs", "docs/change.md", new)

    assert old_chunks[0].source_id == new_chunks[0].source_id
    assert old_chunks[0].source_version_key != new_chunks[0].source_version_key
    assert {chunk.chunk_id for chunk in old_chunks}.isdisjoint(
        {chunk.chunk_id for chunk in new_chunks}
    )


def test_same_path_under_different_owner_or_namespace_has_different_source_ids():
    owner_a = make_index_source_id("owner-a", "docs", "docs/same.md")
    owner_b = make_index_source_id("owner-b", "docs", "docs/same.md")
    namespace_b = make_index_source_id("owner-a", "research", "docs/same.md")

    assert owner_a != owner_b
    assert owner_a != namespace_b


def test_source_version_key_is_deterministic_for_source_and_content_hash():
    source_id = make_index_source_id("owner-a", "docs", "docs/coexist.md")
    version = make_source_version_key(source_id, "source-hash")
    repeat = make_source_version_key(source_id, "source-hash")
    changed = make_source_version_key(source_id, "changed-source-hash")
    staged = build_source_chunks("owner-a", "docs", "docs/coexist.md", "staged body\n")

    assert version == repeat
    assert version != changed
    assert staged[0].source_id == source_id
    assert staged[0].source_version_key not in {version, source_id}
    assert staged[0].chunk_id not in {version, source_id}


def test_defaults_are_frozen_for_phase_three_chunking():
    assert DEFAULT_TARGET_CHARS == 3500
    assert DEFAULT_MAX_CHARS == 5000
    assert DEFAULT_OVERLAP_LINES == 3


def test_single_oversized_line_is_split_below_max_chars():
    blob = "x" * (DEFAULT_MAX_CHARS * 4)
    chunks = build_source_chunks("owner", "business", "business/blob.json", blob)

    assert len(chunks) > 1
    assert all(len(chunk.embedding_text) <= DEFAULT_MAX_CHARS for chunk in chunks)
    reconstructed = "".join(non_overlap_lines(chunks))
    assert reconstructed.replace("\n", "") == blob
