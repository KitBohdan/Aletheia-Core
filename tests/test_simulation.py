from types import MethodType

from vct.robodog.dog_bot_brain import RoboDogBrain
from vct.simulation.dog_env import DogEnv


def test_dog_env_passes_state_to_brain():
    brain = RoboDogBrain(cfg_path="vct/config.yaml", simulate=True)
    env = DogEnv(seed=7)

    captured = {}
    original_decide = brain.policy.decide

    def spy(self, action, inputs):
        captured["mood"] = inputs.mood
        captured["fatigue"] = inputs.fatigue
        return original_decide(action, inputs)

    brain.policy.decide = MethodType(spy, brain.policy)

    first_state = env.observe()
    outcome = env.run_brain_step(
        brain,
        "сидіти",
        confidence=0.9,
        reward_bias=0.7,
    )

    assert captured["mood"] == first_state["mood"]
    assert captured["fatigue"] == first_state["fatigue"]
    assert "brain" in outcome and "state" in outcome
    assert -1.0 <= outcome["state"]["mood"] <= 1.0
    assert 0.0 <= outcome["state"]["fatigue"] <= 1.0


def test_dog_env_episode_success_rate():
    brain = RoboDogBrain(cfg_path="vct/config.yaml", simulate=True)
    env = DogEnv(seed=21)
    env.reset()
    commands = ["сидіти", "голос", "лежати", "сидіти"]

    results = env.simulate_commands(brain, commands, confidence=0.85, reward_bias=0.6)

    assert "success_rate" in results
    assert 0.0 <= results["success_rate"] <= 1.0
    assert len(results["history"]) == len(commands)
    assert "state" in results["history"][-1]
