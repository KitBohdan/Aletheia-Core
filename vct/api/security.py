"""Security helpers for protecting FastAPI endpoints."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
_api_key_value = os.getenv("VCT_API_KEY")
_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


class APIKeyAuthError(HTTPException):
    """Raised when the provided API key is missing or invalid."""

    def __init__(self, detail: str = "Invalid or missing API key") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "API-Key"},
        )


def _get_configured_key() -> str:
    if not _api_key_value:
        raise RuntimeError(
            "API key authentication is enabled but no VCT_API_KEY environment "
            "variable was provided."
        )
    return _api_key_value


def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
    """Validate that the incoming request provides the expected API key."""

    expected = _get_configured_key()
    if api_key != expected:
        raise APIKeyAuthError()
    return expected
