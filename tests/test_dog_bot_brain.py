from pathlib import Path

from vct.robodog.dog_bot_brain import RoboDogBrain


def _write_cfg(tmp_path: Path, content: str) -> Path:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(content, encoding="utf-8")
    return cfg_path


def test_action_from_text_supports_synonyms(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        """
commands_map:
  "сидіти": SIT
  "сідай": SIT
  "sit": SIT
  "до_мене": COME
  "stay": STAY
reward_triggers: {}
        """.strip(),
    )
    brain = RoboDogBrain(str(cfg_path), simulate=True)

    assert brain._action_from_text("Сідай, будь ласка") == "SIT"
    assert brain._action_from_text("Do me a favour and sit down") == "SIT"
    assert brain._action_from_text("Песику, до мене!") == "COME"
    assert brain._action_from_text("Please stay here") == "STAY"


def test_action_from_text_normalizes_delimiters(tmp_path: Path) -> None:
    cfg_path = _write_cfg(
        tmp_path,
        """
commands_map:
  "лягай": LIE_DOWN
  "lie-down": LIE_DOWN
reward_triggers: {}
        """.strip(),
    )
    brain = RoboDogBrain(str(cfg_path), simulate=True)

    assert brain._action_from_text("Лягай негайно") == "LIE_DOWN"
    assert brain._action_from_text("Time to lie down now") == "LIE_DOWN"
