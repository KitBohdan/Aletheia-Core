from vct.robodog.dog_bot_brain import RoboDogBrain

def test_brain_handles_text_and_rewards():
    brain = RoboDogBrain(cfg_path="vct/config.yaml", simulate=True)
    out = brain.handle_command("сидіти", confidence=0.9, reward_bias=0.7, mood=0.2)
    assert out["action"] == "SIT"
    assert isinstance(out["score"], float)
    assert out["rewarded"] in (True, False)

def test_brain_from_wav_rulebased():
    brain = RoboDogBrain(cfg_path="vct/config.yaml", simulate=True)
    out = brain.run_once_from_wav("data/examples/commands/sydity.wav")
    assert out["action"] in ("SIT","NONE")