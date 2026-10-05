"""memory-v3: Next-generation brain-inspired persistent memory for AI coding assistants."""

__version__ = "1.0.0"
__author__ = "Sean Pembroke"

__all__ = ["get_embedder", "get_version_info"]


def __getattr__(name: str):
    if name == "get_embedder":
        from .providers import get_embedder

        return get_embedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_version_info() -> dict:
    """Return version metadata for diagnostics."""
    return {
        "name": "memory-v3",
        "version": __version__,
        "author": __author__,
        "description": "Brain-inspired persistent memory with governance layers, "
        "sensory gating, and multi-graph knowledge representation.",
    }
