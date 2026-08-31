"""Task-specific runtime adapters registered by the outer inference router."""

from .refseg import RefSegAdapter

__all__ = ("RefSegAdapter",)
