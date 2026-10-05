"""
memory-v3 -- Community Detection
Leiden community detection and centroid computation for graph layers.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import networkx as nx
import numpy as np


def detect_communities(G: nx.DiGraph, resolution: float = 1.0) -> dict:
    """
    Run Leiden community detection on a graph layer.

    Uses python-igraph + leidenalg for high-quality partitioning.
    Falls back to networkx greedy modularity if igraph is unavailable.

    Args:
        G: The networkx DiGraph to partition.
        resolution: Higher values produce more, smaller communities.

    Returns dict:
        {
            "communities": {community_id: [node_names]},
            "modularity": float,
            "num_communities": int,
            "algorithm": str,
        }
    """
    if G.number_of_nodes() == 0:
        return {"communities": {}, "modularity": 0.0, "num_communities": 0, "algorithm": "none"}

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    try:
        import igraph as ig
        import leidenalg

        # Convert networkx -> igraph
        node_list = sorted(G_undirected.nodes, key=str)
        node_index = {n: i for i, n in enumerate(node_list)}

        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(node_list))
        ig_graph.vs["name"] = node_list

        edges = []
        for u, v in G_undirected.edges():
            edges.append((node_index[u], node_index[v]))
        ig_graph.add_edges(edges)

        # Run Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
        )

        communities = {}
        for comm_id, members in enumerate(partition):
            communities[comm_id] = sorted((node_list[i] for i in members), key=str)

        return {
            "communities": communities,
            "modularity": round(partition.modularity, 4),
            "num_communities": len(communities),
            "algorithm": "leiden",
        }

    except ImportError:
        # Fallback: networkx greedy modularity on undirected graph
        from networkx.algorithms.community import greedy_modularity_communities

        comms = list(greedy_modularity_communities(G_undirected, resolution=resolution))

        communities = {}
        for comm_id, members in enumerate(comms):
            communities[comm_id] = sorted(members, key=str)

        # Compute modularity
        try:
            mod = nx.algorithms.community.quality.modularity(G_undirected, comms)
        except Exception:
            mod = 0.0

        return {
            "communities": communities,
            "modularity": round(mod, 4),
            "num_communities": len(communities),
            "algorithm": "greedy_modularity",
        }


def compute_centroids(
    conn: sqlite3.Connection,
    G: nx.DiGraph,
    embeddings_module,
) -> list[dict]:
    """
    Compute centroid embeddings for each community and store in the database.

    For each community detected in G:
      1. Collect embeddings of all member nodes (from node attrs or by embedding node names).
      2. Compute the mean embedding as the centroid.
      3. Store in cluster_centroids table.

    Args:
        conn: SQLite connection with a cluster_centroids table.
        G: The graph to partition.
        embeddings_module: Module with embed_text(str) -> list[float] and embed_batch(list[str]).

    Returns list of dicts: [{community_id, num_members, centroid_dim}]
    """
    # Ensure the table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cluster_centroids (
            community_id INTEGER PRIMARY KEY,
            centroid BLOB NOT NULL,
            num_members INTEGER DEFAULT 0,
            member_nodes TEXT DEFAULT '[]',
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()

    detection = detect_communities(G)
    communities = detection["communities"]
    results = []

    for comm_id, members in communities.items():
        if not members:
            continue

        # Collect embeddings for members
        embeddings = []
        for member in members:
            node_data = G.nodes.get(member, {})
            emb = node_data.get("embedding")
            if emb is not None:
                embeddings.append(np.array(emb))

        # If fewer than half of members have stored embeddings, embed by name
        if len(embeddings) < len(members) / 2:
            try:
                texts = [str(m) for m in members]
                batch_embs = embeddings_module.embed_batch(texts)
                embeddings = [np.array(e) for e in batch_embs]
            except Exception:
                # Embed one at a time as fallback
                embeddings = []
                for m in members:
                    try:
                        emb = embeddings_module.embed_text(str(m))
                        embeddings.append(np.array(emb))
                    except Exception:
                        continue

        if not embeddings:
            continue

        # Compute centroid as mean embedding
        centroid = np.mean(embeddings, axis=0).astype(np.float32)
        centroid_bytes = centroid.tobytes()
        ts = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """INSERT OR REPLACE INTO cluster_centroids
               (community_id, centroid, num_members, member_nodes, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (comm_id, centroid_bytes, len(members), json.dumps(members), ts),
        )

        results.append({
            "community_id": comm_id,
            "num_members": len(members),
            "centroid_dim": len(centroid),
        })

    conn.commit()
    return results


def get_community_for_query(
    query_embedding: list[float] | np.ndarray,
    conn: sqlite3.Connection,
    top_n: int = 3,
) -> list[int]:
    """
    Find the top-N communities whose centroids are most similar to the query embedding.

    Args:
        query_embedding: The query vector.
        conn: SQLite connection with cluster_centroids table.
        top_n: Number of community IDs to return.

    Returns list of community_id integers, ordered by descending cosine similarity.
    """
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm < 1e-9:
        return []

    columns = {row[1] for row in conn.execute("PRAGMA table_info(cluster_centroids)").fetchall()}
    if {"community_id", "centroid"} <= columns:
        cursor = conn.execute("SELECT community_id, centroid FROM cluster_centroids")
    else:
        cursor = conn.execute(
            "SELECT cluster_id AS community_id, centroid_embedding AS centroid "
            "FROM cluster_centroids"
        )
    scored = []

    for comm_id, centroid_bytes in cursor:
        centroid = np.frombuffer(centroid_bytes, dtype=np.float32)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-9:
            continue

        similarity = float(np.dot(query_vec, centroid) / (query_norm * centroid_norm))
        scored.append((comm_id, similarity))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [comm_id for comm_id, _ in scored[:top_n]]
