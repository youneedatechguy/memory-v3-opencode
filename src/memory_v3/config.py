"""
memory-v3 -- Central Configuration
ALL settings via MEMORY_V3_* environment variables with sensible defaults.
Uses dataclasses for structured config and a cached get_config() accessor.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Governance Layer Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerConfig:
    """Configuration for a single governance layer."""
    name: str
    layer_id: int
    decay_rate: float
    activation_floor: float
    fademem_beta: float          # FadeMem recency weight (N/A stored as 0.0)
    archive_threshold: float     # 0.0 = never archive
    min_archive_age_days: int    # minimum age before archiving is considered
    archivable: bool = True      # False = NEVER archive regardless of scores


# The four governance layers -- immutable by design
LAYER_CONSTITUTIONAL = LayerConfig(
    name="Constitutional",
    layer_id=1,
    decay_rate=0.0,
    activation_floor=5.0,
    fademem_beta=0.0,            # N/A -- no decay
    archive_threshold=0.0,
    min_archive_age_days=0,
    archivable=False,
)

LAYER_LEGISLATIVE = LayerConfig(
    name="Legislative",
    layer_id=2,
    decay_rate=0.01,
    activation_floor=2.0,
    fademem_beta=0.5,
    archive_threshold=0.0,
    min_archive_age_days=0,
    archivable=False,
)

LAYER_FACTUAL = LayerConfig(
    name="Factual",
    layer_id=3,
    decay_rate=0.10,
    activation_floor=-0.5,
    fademem_beta=0.8,
    archive_threshold=0.1,
    min_archive_age_days=30,
    archivable=True,
)

LAYER_EPHEMERAL = LayerConfig(
    name="Ephemeral",
    layer_id=4,
    decay_rate=0.25,
    activation_floor=-2.0,
    fademem_beta=1.2,
    archive_threshold=0.15,
    min_archive_age_days=14,
    archivable=True,
)

GOVERNANCE_LAYERS: dict[int, LayerConfig] = {
    1: LAYER_CONSTITUTIONAL,
    2: LAYER_LEGISLATIVE,
    3: LAYER_FACTUAL,
    4: LAYER_EPHEMERAL,
}


# ---------------------------------------------------------------------------
# Main Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Complete memory-v3 configuration. Populated from env vars once."""

    # --- Paths ---
    db_path: str = ""
    vault_path: str = ""
    graph_dir: str = ""
    cache_dir: str = ""

    # --- Model ---
    embed_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:3b"
    embed_dim: int = 768
    embed_provider: str = "ollama"   # MEMORY_V3_EMBED_PROVIDER
    llm_provider: str = "ollama"     # MEMORY_V3_LLM_PROVIDER

    # --- ACT-R cognitive architecture constants ---
    actr_decay_d: float = 0.5
    actr_s_max: float = 1.6
    actr_noise_s: float = 0.25
    actr_retrieval_tau: float = -0.5

    # --- FadeMem importance weights ---
    # alpha=recency, beta=frequency, gamma=relevance, delta=surprise
    fademem_alpha: float = 0.30
    fademem_beta: float = 0.25
    fademem_gamma: float = 0.25
    fademem_delta: float = 0.20   # surprise component

    # --- Fan effect ---
    fan_delta: float = 0.3

    # --- Sensory filter ---
    sensory_min_length: int = 20
    sensory_dedup_threshold: float = 0.95

    # --- Retrieval ---
    rrf_k: int = 60
    stage1_max_candidates: int = 500
    stage1_community_top: int = 3
    zettel_link_threshold: float = 0.5

    # --- Async processing ---
    async_enabled: bool = True
    async_workers: int = 2
    async_batch_size: int = 10

    # --- Consolidation ---
    consolidation_merge_threshold: float = 0.85
    consolidation_min_group_size: int = 3

    # --- HaluMem (confidence decay for unverified memories) ---
    halumem_confidence_decay_days: int = 90
    halumem_confidence_decay_rate: float = 0.1

    # --- Feature flags ---
    enable_multigraph: bool = True
    enable_sensory: bool = True
    enable_zettelkasten: bool = True
    enable_surprise: bool = True
    enable_fan_effect: bool = True
    enable_rl_decider: bool = False
    enable_three_stage: bool = True
    enable_halumem: bool = True
    enable_hierarchy: bool = True
    enable_consolidation: bool = True
    enable_async: bool = True
    enable_structure_tags: bool = True

    # --- Multi-agent authority chain ---
    authority_chain: dict = field(default_factory=lambda: {
        "human": 1,
        "primary_agent": 2,
        "coordinator": 3,
        "worker": 4,
        "indexer": 5,
        "extractor": 6,
    })

    # --- Governance layers (immutable, not from env) ---
    governance_layers: dict = field(default_factory=lambda: dict(GOVERNANCE_LAYERS))


def _env(key: str, default: str = "") -> str:
    """Read a MEMORY_V3_* env var."""
    return os.environ.get(f"MEMORY_V3_{key}", default)


def _env_bool(key: str, default: bool) -> bool:
    """Read a MEMORY_V3_ENABLE_* env var as bool."""
    val = os.environ.get(f"MEMORY_V3_ENABLE_{key}", "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    """Read a MEMORY_V3_* env var as float."""
    val = os.environ.get(f"MEMORY_V3_{key}", "")
    if not val:
        return default
    return float(val)


def _env_int(key: str, default: int) -> int:
    """Read a MEMORY_V3_* env var as int."""
    val = os.environ.get(f"MEMORY_V3_{key}", "")
    if not val:
        return default
    return int(val)


def _build_config() -> Config:
    """Build a Config from environment variables."""
    home = str(Path.home())
    default_base = os.path.join(home, ".memory-v3")

    # Parse authority chain from JSON env var if provided
    authority_chain_raw = os.environ.get("MEMORY_V3_AUTHORITY_CHAIN", "")
    if authority_chain_raw:
        authority_chain = json.loads(authority_chain_raw)
    else:
        authority_chain = {
            "human": 1,
            "primary_agent": 2,
            "coordinator": 3,
            "worker": 4,
            "indexer": 5,
            "extractor": 6,
        }

    return Config(
        # Paths
        db_path=_env("DB", os.path.join(default_base, "memory.db")),
        vault_path=_env("VAULT", ""),
        graph_dir=_env("GRAPH_DIR", os.path.join(default_base, "graphs")),
        cache_dir=_env("CACHE", os.path.join(default_base, "cache")),

        # Model
        embed_model=_env("EMBED_MODEL", "nomic-embed-text"),
        llm_model=_env("LLM_MODEL", "qwen2.5:3b"),
        embed_dim=_env_int("EMBED_DIM", 768),
        embed_provider=_env("EMBED_PROVIDER", "ollama"),
        llm_provider=_env("LLM_PROVIDER", "ollama"),

        # ACT-R
        actr_decay_d=_env_float("ACTR_DECAY_D", 0.5),
        actr_s_max=_env_float("ACTR_S_MAX", 1.6),
        actr_noise_s=_env_float("ACTR_NOISE_S", 0.25),
        actr_retrieval_tau=_env_float("ACTR_RETRIEVAL_TAU", -0.5),

        # FadeMem
        fademem_alpha=_env_float("FADEMEM_ALPHA", 0.30),
        fademem_beta=_env_float("FADEMEM_BETA", 0.25),
        fademem_gamma=_env_float("FADEMEM_GAMMA", 0.25),
        fademem_delta=_env_float("FADEMEM_DELTA", 0.20),

        # Fan effect
        fan_delta=_env_float("FAN_DELTA", 0.3),

        # Sensory filter
        sensory_min_length=_env_int("SENSORY_MIN_LENGTH", 20),
        sensory_dedup_threshold=_env_float("SENSORY_DEDUP_THRESHOLD", 0.95),

        # Retrieval
        rrf_k=_env_int("RRF_K", 60),
        stage1_max_candidates=_env_int("STAGE1_MAX_CANDIDATES", 500),
        stage1_community_top=_env_int("STAGE1_COMMUNITY_TOP", 3),
        zettel_link_threshold=_env_float("ZETTEL_LINK_THRESHOLD", 0.5),

        # Async
        async_enabled=_env_bool("ASYNC", True),
        async_workers=_env_int("ASYNC_WORKERS", 2),
        async_batch_size=_env_int("ASYNC_BATCH_SIZE", 10),

        # Consolidation
        consolidation_merge_threshold=_env_float("CONSOLIDATION_MERGE_THRESHOLD", 0.85),
        consolidation_min_group_size=_env_int("CONSOLIDATION_MIN_GROUP_SIZE", 3),

        # HaluMem
        halumem_confidence_decay_days=_env_int("HALUMEM_CONFIDENCE_DECAY_DAYS", 90),
        halumem_confidence_decay_rate=_env_float("HALUMEM_CONFIDENCE_DECAY_RATE", 0.1),

        # Feature flags
        enable_multigraph=_env_bool("MULTIGRAPH", True),
        enable_sensory=_env_bool("SENSORY", True),
        enable_zettelkasten=_env_bool("ZETTELKASTEN", True),
        enable_surprise=_env_bool("SURPRISE", True),
        enable_fan_effect=_env_bool("FAN_EFFECT", True),
        enable_rl_decider=_env_bool("RL_DECIDER", False),
        enable_three_stage=_env_bool("THREE_STAGE", True),
        enable_halumem=_env_bool("HALUMEM", True),
        enable_hierarchy=_env_bool("HIERARCHY", True),
        enable_consolidation=_env_bool("CONSOLIDATION", True),
        enable_async=_env_bool("ASYNC", True),
        enable_structure_tags=_env_bool("STRUCTURE_TAGS", True),

        # Multi-agent
        authority_chain=authority_chain,
    )


# ---------------------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------------------

_config: Optional[Config] = None


def get_config() -> Config:
    """Get the cached configuration. Reads env vars only on first call."""
    global _config
    if _config is None:
        _config = _build_config()
    return _config


def reset_config():
    """Reset cached config (useful for testing)."""
    global _config
    _config = None


def get_layer_config(layer_id: int) -> LayerConfig:
    """Get the LayerConfig for a governance layer (1-4)."""
    if layer_id not in GOVERNANCE_LAYERS:
        raise ValueError(f"Invalid governance layer: {layer_id}. Must be 1-4.")
    return GOVERNANCE_LAYERS[layer_id]
