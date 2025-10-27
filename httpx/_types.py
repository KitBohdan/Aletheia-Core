"""Typing aliases used at runtime only for attribute lookups."""
from __future__ import annotations

# The real httpx library exposes a collection of typing aliases.  For the purposes of
# the kata environment we only need placeholder objects so that attribute accesses from
# Starlette's TestClient succeed.  Using ``object`` keeps runtime behaviour simple while
# remaining type-friendly for static analysis.

URLTypes = object
RequestContent = object
RequestFiles = object
QueryParamTypes = object
HeaderTypes = object
CookieTypes = object
AuthTypes = object
TimeoutTypes = object
