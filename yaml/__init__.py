"""A tiny YAML loader sufficient for the project's configuration fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

__all__ = ["safe_load", "safe_dump"]


@dataclass
class _StackFrame:
    indent: int
    container: dict[str, Any]


def _parse_scalar(token: str) -> Any:
    if token == "":
        return ""
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in token for ch in (".", "e", "E")):
            return float(token)
        return int(token)
    except ValueError:
        pass
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        parts: list[str] = []
        depth = 0
        start = 0
        for idx, char in enumerate(inner):
            if char == "," and depth == 0:
                parts.append(inner[start:idx].strip())
                start = idx + 1
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
        parts.append(inner[start:].strip())
        for part in parts:
            if not part:
                continue
            key, value = part.split(":", 1)
            result[key.strip()] = _parse_scalar(value.strip())
        return result
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    return token


def safe_load(stream: Any) -> dict[str, Any]:
    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = stream
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        raise TypeError("safe_load() expects a string or a text IO object")

    root: dict[str, Any] = {}
    stack: list[_StackFrame] = [_StackFrame(indent=-1, container=root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1].indent:
            stack.pop()
        current = stack[-1].container
        if stripped.endswith(":"):
            key = stripped[:-1].strip()
            new_map: dict[str, Any] = {}
            current[key] = new_map
            stack.append(_StackFrame(indent=indent, container=new_map))
            continue
        if ":" not in stripped:
            raise ValueError(f"Unable to parse line: {raw_line!r}")
        key, value = stripped.split(":", 1)
        current[key.strip()] = _parse_scalar(value.strip())

    return root


def safe_dump(data: dict[str, Any], *, allow_unicode: bool = True, sort_keys: bool = True) -> str:
    """Serialize a mapping to a YAML-like string."""

    return json.dumps(
        data,
        ensure_ascii=not allow_unicode,
        sort_keys=sort_keys,
        indent=2,
    )
