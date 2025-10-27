import builtins
from pathlib import Path

import pytest

from vct.configuration import (
    RoboDogSettings,
    _read_config,
    _write_config,
    apply_key_path,
    parse_typed_value,
)


def write_config(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_configuration_and_normalization(tmp_path: Path) -> None:
    cfg_path = write_config(
        tmp_path / "config.yaml",
        """
reward_cooldown_s: 1.5
commands_map:
  сидіти: sit
reward_triggers:
  sit: true
weights:
  confidence: 0.9
        """.strip(),
    )
    settings = RoboDogSettings.load(cfg_path)
    assert settings.reward_cooldown_s == 1.5
    assert settings.commands_map["сидіти"] == "SIT"
    assert settings.reward_triggers["SIT"] is True
    assert settings.policy_config == {"confidence": 0.9}


def test_invalid_command_mapping_raises(tmp_path: Path) -> None:
    cfg_path = write_config(
        tmp_path / "config.yaml",
        "commands_map:\n  '': SIT\n",
    )
    with pytest.raises(ValueError):
        RoboDogSettings.load(cfg_path)


def test_apply_key_path_updates(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.yaml", "reward_cooldown_s: 2\n")
    settings = RoboDogSettings.load(cfg_path)
    updated = apply_key_path(settings, ["environment_context", "complexity"], 0.7)
    assert updated.environment_context["complexity"] == 0.7


def test_parse_typed_value() -> None:
    assert parse_typed_value("true", "bool") is True
    assert parse_typed_value("3.14", "float") == pytest.approx(3.14)
    assert parse_typed_value("42", "int") == 42
    assert parse_typed_value('{"a": 1}', "json") == {"a": 1}
    with pytest.raises(ValueError):
        parse_typed_value("maybe", "bool")


def test_parse_typed_value_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported value type"):
        parse_typed_value("hello", "uuid")


def test_load_missing_configuration_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RoboDogSettings.load(tmp_path / "missing.yaml")


def test_load_invalid_weights_raises(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.yaml", "weights: invalid\n")
    with pytest.raises(TypeError, match="weights must be a mapping"):
        RoboDogSettings.load(cfg_path)


def test_apply_key_path_empty_segments(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.yaml", "reward_cooldown_s: 2\n")
    settings = RoboDogSettings.load(cfg_path)
    with pytest.raises(ValueError):
        apply_key_path(settings, [], 3)


def test_read_config_unsupported_format(tmp_path: Path) -> None:
    cfg_path = write_config(tmp_path / "config.ini", "[section]\nvalue=1\n")
    with pytest.raises(ValueError, match="Unsupported configuration format"):
        _read_config(cfg_path)


def test_write_config_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported configuration format"):
        _write_config(tmp_path / "config.ini", {"a": 1})


def test_write_config_requires_tomli_w_for_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "tomli_w":
            raise ImportError("No module named 'tomli_w'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="tomli-w"):
        _write_config(tmp_path / "config.toml", {"value": 1})
