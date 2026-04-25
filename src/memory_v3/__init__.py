"""memory-v3: Next-generation brain-inspired persistent memory for AI coding assistants."""

__version__ = "1.0.0"
__author__ = "Sean Pembroke"

from .providers import get_embedder  # noqa: F401  re-export for public API


def get_version_info() -> dict:
    """Return version metadata for diagnostics."""
    return {
        "name": "memory-v3",
        "version": __version__,
        "author": __author__,
        "description": "Brain-inspired persistent memory with governance layers, "
                       "sensory gating, and multi-graph knowledge representation.",
    }
