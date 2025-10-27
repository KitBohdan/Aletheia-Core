from pathlib import Path

import pytest

from vct.configuration import RoboDogSettings, apply_key_path, parse_typed_value


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
        """.strip()
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
    assert parse_typed_value("{\"a\": 1}", "json") == {"a": 1}
    with pytest.raises(ValueError):
        parse_typed_value("maybe", "bool")


