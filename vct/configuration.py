from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import json

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class RoboDogSettings(BaseModel):
    """Validated configuration for the RoboDog brain."""

    latency_budget_ms: int = Field(300, ge=0)
    reward_cooldown_s: float = Field(3.0, ge=0.0)
    weights: Dict[str, float] = Field(default_factory=dict)
    behavior_policy: Dict[str, Any] | None = None
    policy: Dict[str, Any] | None = None
    commands_map: Dict[str, str] = Field(default_factory=dict)
    reward_triggers: Dict[str, bool] = Field(default_factory=dict)
    environment_context: Dict[str, float] = Field(default_factory=dict)
    mood_initial: str | None = None

    @field_validator("weights", mode="before")
    @classmethod
    def _ensure_float_mapping(cls, value: Any) -> Dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return {str(k): float(v) for k, v in value.items()}
        raise TypeError("weights must be a mapping of feature name to float weight")

    @field_validator("commands_map", mode="before")
    @classmethod
    def _normalize_commands(cls, value: Any) -> Dict[str, str]:
        if value is None:
            return {}
        if isinstance(value, dict):
            normalized = {}
            for k, v in value.items():
                raw_key = str(k)
                key = raw_key.strip()
                cleaned = key.strip("'\"")
                if not cleaned:
                    raise ValueError("Command map keys must be non-empty strings")
                normalized[cleaned.lower()] = str(v).strip().upper() or "NONE"
            return normalized
        raise TypeError("commands_map must be a mapping of phrase to action")

    @field_validator("reward_triggers", mode="before")
    @classmethod
    def _normalize_triggers(cls, value: Any) -> Dict[str, bool]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(k).strip().upper(): bool(v) for k, v in value.items()}
        raise TypeError("reward_triggers must be a mapping of action to boolean")

    @field_validator("environment_context", mode="before")
    @classmethod
    def _normalize_context(cls, value: Any) -> Dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items()}
        raise TypeError("environment_context must be a mapping of context values")

    @property
    def policy_config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for the behavior policy."""

        if self.behavior_policy is not None:
            return self.behavior_policy
        if self.policy is not None:
            return self.policy
        if self.weights:
            return self.weights
        return {}

    @classmethod
    def load(cls, path: str | Path) -> "RoboDogSettings":
        payload = _read_config(path)
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    def save(self, path: str | Path) -> None:
        _write_config(path, self.model_dump(mode="json"))

    def updated(self, updates: Dict[str, Any]) -> "RoboDogSettings":
        data = self.model_dump()
        data.update(updates)
        return type(self).model_validate(data)


def _read_config(path: str | Path) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path_obj}")
    suffix = path_obj.suffix.lower()
    text = path_obj.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml", ""}:
        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        import tomllib

        return tomllib.loads(text)
    raise ValueError(f"Unsupported configuration format: {suffix}")


def _write_config(path: str | Path, payload: Dict[str, Any]) -> None:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix in {".yaml", ".yml", ""}:
        serialized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
    elif suffix == ".json":
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    elif suffix == ".toml":
        try:
            import tomli_w
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Writing TOML configurations requires the 'tomli-w' package"
            ) from exc

        serialized = tomli_w.dumps(payload)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")
    path_obj.write_text(serialized, encoding="utf-8")


def apply_key_path(
    settings: RoboDogSettings,
    key_path: Iterable[str],
    value: Any,
) -> RoboDogSettings:
    data: Dict[str, Any] = settings.model_dump(mode="json")
    target: MutableMapping[str, Any] = data
    segments: Tuple[str, ...] = tuple(key_path)
    if not segments:
        raise ValueError("Key path cannot be empty")
    for part in segments[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        node = target[part]
        if not isinstance(node, dict):
            raise TypeError(f"Cannot set nested key on non-mapping value at '{part}'")
        target = node
    target[segments[-1]] = value
    return RoboDogSettings.model_validate(data)


def parse_typed_value(raw: str, value_type: str) -> Any:
    if value_type == "str":
        return raw
    if value_type == "int":
        return int(raw)
    if value_type == "float":
        return float(raw)
    if value_type == "bool":
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value from '{raw}'")
    if value_type == "json":
        return json.loads(raw)
    raise ValueError(f"Unsupported value type: {value_type}")

