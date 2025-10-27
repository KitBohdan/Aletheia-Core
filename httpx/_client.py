"""Lightweight stand-ins for ``httpx._client`` symbols used by Starlette's TestClient."""
from __future__ import annotations


class UseClientDefault:
    """Sentinel type used to mirror the real httpx API surface."""

    def __repr__(self) -> str:  # pragma: no cover - trivial representation
        return "UseClientDefault()"


USE_CLIENT_DEFAULT = UseClientDefault()
