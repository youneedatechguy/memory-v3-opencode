"""Deprecated knowledge graph builder compatibility entry points."""

from __future__ import annotations

import json
import re
from pathlib import Path

GRAPH_LAYERS = ("semantic", "temporal", "causal", "entity")
GRAPH_ARTIFACTS = {
    "semantic": "semantic.pkl",
    "temporal": "temporal.pkl",
    "causal": "causal.pkl",
    "entity": "entity.pkl",
    "communities": "communities.json",
    "router_cache": "router_cache.json",
}
GRAPH_REBUILD_TIMESTAMP = "1970-01-01T00:00:00+00:00"

EXTRACT_PROMPT = """Extract entities and relationships from this text.

Return JSON with two lists:
1. "entities": each with name, type, properties (optional dict)
2. "relationships": each with source, target, edge_type, description (optional)

If nothing extractable, return {"entities": [], "relationships": []}

Text:
{text}

JSON:"""


_DISABLED_RESULT = {
    "status": "disabled",
    "deprecated": True,
    "code": "knowledge_graph_disabled",
    "error": (
        "Legacy knowledge graph building is disabled; active memory-v3 operation is graph-free."
    ),
    "files_processed": 0,
    "files_skipped": 0,
    "files_errored": 0,
    "total_entities": 0,
    "total_relationships": 0,
    "per_layer": {},
}


def _disabled_stats(backend: str | None = None) -> dict:
    result = dict(_DISABLED_RESULT)
    if backend is not None:
        result["backend"] = backend
    return result


def _parse_extraction(content: str) -> dict:
    try:
        json_match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(json_match.group() if json_match else content)
        valid_entities = []
        for entity in data.get("entities", []):
            if isinstance(entity, dict) and entity.get("name"):
                valid_entities.append(
                    {
                        "name": str(entity["name"]).strip(),
                        "type": str(entity.get("type", "concept")).strip().lower(),
                        "properties": entity.get("properties", {}),
                    }
                )
        valid_relationships = []
        for relationship in data.get("relationships", []):
            if (
                isinstance(relationship, dict)
                and relationship.get("source")
                and relationship.get("target")
            ):
                valid_relationships.append(
                    {
                        "source": str(relationship["source"]).strip(),
                        "target": str(relationship["target"]).strip(),
                        "edge_type": str(relationship.get("edge_type", "related_to"))
                        .strip()
                        .lower(),
                        "description": str(relationship.get("description", "")),
                        "confidence": float(relationship.get("confidence", 0.8)),
                    }
                )
        return {"entities": valid_entities, "relationships": valid_relationships}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"entities": [], "relationships": [], "parse_error": content[:200]}


def _extract_ollama(text: str, model: str) -> dict:
    return {"entities": [], "relationships": []}


async def _extract_anthropic_single(client, text: str, semaphore) -> dict:
    return {"entities": [], "relationships": []}


async def _extract_anthropic_batch(
    client,
    file_texts: list[tuple[str, str, str]],
    semaphore,
) -> list[tuple[str, str, dict]]:
    return [
        (relative_path, content_hash, {"entities": [], "relationships": []})
        for relative_path, content_hash, _ in file_texts
    ]


def build_knowledge_graph(
    vault_path: str | None = None,
    force: bool = False,
    conn=None,
    backend: str = "ollama",
) -> dict:
    return _disabled_stats(backend)


def rebuild_graph_artifacts_from_active_generations(
    conn,
    graph_dir: str | Path,
    *,
    extraction_fn=None,
    publish_hook=None,
) -> dict:
    result = _disabled_stats()
    result["artifact_counts"] = {}
    result["source_rows"] = 0
    return result


def _default_active_generation_extractor(content: str) -> dict:
    return {"entities": [], "relationships": []}


def build_graph_rebuild_communities(graph_dir: str | Path) -> dict:
    return {
        "communities": {},
        "router_cache": {"layers": list(GRAPH_LAYERS), "layer_counts": {}},
        "centroids": [],
    }


def _deterministic_centroid(members: list[str]) -> bytes:
    return b""


def _publish_graph_artifacts(temp_dir: Path, target_dir: Path) -> None:
    return None


def _graph_rebuild_stats(temp_dir: str, artifact_paths: dict) -> dict:
    result = _disabled_stats()
    result["temp_dir"] = temp_dir
    result["artifact_paths"] = artifact_paths
    result["source_rows"] = 0
    result["artifact_counts"] = {}
    return result


def _accumulate_layer_stats(stats: dict, layer_stats: dict) -> None:
    return None


def _artifact_counts(temp_dir: Path, community_state: dict) -> dict:
    return {}


def _sanitized_graph_error(exc: Exception) -> dict:
    return {
        "error_category": "knowledge_graph_disabled",
        "exception_class": exc.__class__.__name__,
    }


def _build_kg_ollama(
    vault_path: str | None = None,
    force: bool = False,
    conn=None,
) -> dict:
    return _disabled_stats("ollama")


async def _build_kg_anthropic(
    vault_path: str | None = None,
    force: bool = False,
    conn=None,
) -> dict:
    return _disabled_stats("anthropic")


def _process_extraction(
    conn,
    gm,
    extracted: dict,
    rel_path: str,
    content_hash: str,
    now_iso: str,
    stats: dict,
):
    stats.setdefault("files_skipped", 0)
    stats["files_skipped"] += 1


def _update_kg_hash(conn, rel_path: str, content_hash: str, timestamp: str, chunk_count: int):
    return None
