import pytest

from memory_v3.config import get_config, reset_config


@pytest.fixture(autouse=True)
def reset_config_singleton():
    reset_config()
    yield
    reset_config()


def test_rollout_gate_defaults_fail_closed():
    cfg = get_config()

    assert cfg.indexing_enabled is False
    assert cfg.vault_auto_extract_enabled is False
    assert cfg.direct_writes_enabled is False
    assert cfg.handoff_sweeper_enabled is False
    assert cfg.handoff_ttl_days == 30
    assert cfg.source_policy_project_stale_days == 90


def test_rollout_gate_env_overrides_and_singleton(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_INDEXING_ENABLED", "true")
    monkeypatch.setenv("MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED", "1")
    monkeypatch.setenv("MEMORY_V3_DIRECT_WRITES_ENABLED", "yes")
    monkeypatch.setenv("MEMORY_V3_HANDOFF_SWEEPER_ENABLED", "on")
    monkeypatch.setenv("MEMORY_V3_HANDOFF_TTL_DAYS", "45")
    monkeypatch.setenv("MEMORY_V3_SOURCE_POLICY_PROJECT_STALE_DAYS", "120")

    cfg = get_config()

    assert cfg.indexing_enabled is True
    assert cfg.vault_auto_extract_enabled is True
    assert cfg.direct_writes_enabled is True
    assert cfg.handoff_sweeper_enabled is True
    assert cfg.handoff_ttl_days == 45
    assert cfg.source_policy_project_stale_days == 120

    monkeypatch.setenv("MEMORY_V3_INDEXING_ENABLED", "false")
    assert get_config().indexing_enabled is True

    reset_config()
    assert get_config().indexing_enabled is False


@pytest.mark.parametrize(
    "env_name",
    [
        "MEMORY_V3_INDEXING_ENABLED",
        "MEMORY_V3_VAULT_AUTO_EXTRACT_ENABLED",
        "MEMORY_V3_DIRECT_WRITES_ENABLED",
        "MEMORY_V3_HANDOFF_SWEEPER_ENABLED",
        "MEMORY_V3_ENABLE_ASYNC",
    ],
)
def test_invalid_boolean_values_fail_clearly(monkeypatch, env_name):
    monkeypatch.setenv(env_name, "definitely")

    with pytest.raises(ValueError, match=f"{env_name} must be a boolean"):
        get_config()


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("MEMORY_V3_HANDOFF_TTL_DAYS", "0"),
        ("MEMORY_V3_HANDOFF_TTL_DAYS", "-1"),
        ("MEMORY_V3_SOURCE_POLICY_PROJECT_STALE_DAYS", "0"),
        ("MEMORY_V3_SOURCE_POLICY_PROJECT_STALE_DAYS", "-1"),
    ],
)
def test_non_positive_day_thresholds_fail_clearly(monkeypatch, env_name, value):
    monkeypatch.setenv(env_name, value)

    with pytest.raises(ValueError, match=f"{env_name} must be a positive integer"):
        get_config()


def test_invalid_integer_values_fail_deterministically(monkeypatch):
    monkeypatch.setenv("MEMORY_V3_HANDOFF_TTL_DAYS", "thirty")

    with pytest.raises(ValueError, match="invalid literal for int"):
        get_config()


def test_removed_lean_envs_have_no_config_attributes_or_validation_effect(monkeypatch):
    removed_envs = {
        "MEMORY_V3_GENERATION_ACTIVATION_ENABLED": "definitely",
        "MEMORY_V3_RETRIEVAL_V2_ENABLED": "definitely",
        "MEMORY_V3_GRAPH_MUTATION_ENABLED": "definitely",
        "MEMORY_V3_LEGACY_AGENT_SYNC_ENABLED": "definitely",
        "MEMORY_V3_SOURCE_POLICY_OLD_RESEARCH_RANK_WEIGHT": "not-a-float",
        "MEMORY_V3_SOURCE_POLICY_PAUSED_PROJECT_RANK_WEIGHT": "not-a-float",
    }
    for env_name, value in removed_envs.items():
        monkeypatch.setenv(env_name, value)

    cfg = get_config()

    assert cfg.indexing_enabled is False
    assert cfg.vault_auto_extract_enabled is False
    assert cfg.direct_writes_enabled is False
    assert cfg.handoff_sweeper_enabled is False
    for attr in [
        "retrieval_v2_enabled",
        "graph_mutation_enabled",
        "legacy_agent_sync_enabled",
        "source_policy_old_research_rank_weight",
        "source_policy_paused_project_rank_weight",
    ]:
        assert not hasattr(cfg, attr)
