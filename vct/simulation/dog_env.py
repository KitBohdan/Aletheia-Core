"""Simple behavioural simulator for the robo-dog brain."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from ..robodog.dog_bot_brain import RoboDogBrain


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class EnvState:
    """Container describing the dog's internal state."""

    fatigue: float = 0.0
    mood: float = 0.0  # -1 .. 1
    reward_history: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {
            "fatigue": self.fatigue,
            "mood": self.mood,
            "reward_hist": self.reward_history,
        }


class DogEnv:
    """Environment that can interact with :class:`RoboDogBrain` in a loop."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self.s = EnvState()

    # ------------------------------------------------------------------
    # Environment dynamics helpers
    # ------------------------------------------------------------------
    def reset(self) -> dict[str, float]:
        """Reset the environment to a neutral internal state."""

        self.s = EnvState()
        return self.observe()

    def observe(self) -> dict[str, float]:
        """Return a shallow copy of the current state."""

        return dict(self.s.as_dict())

    # ------------------------------------------------------------------
    # Simulation logic
    # ------------------------------------------------------------------
    def step(self, action: str, score: float) -> dict[str, Any]:
        """Advance the environment after the brain selects an action."""

        fatigue_penalty = 0.25 * self.s.fatigue
        mood_bonus = 0.1 * self.s.mood
        success_p = 0.5 + 0.4 * score - fatigue_penalty + mood_bonus
        success = self._rng.random() < _clamp(success_p, 0.05, 0.95)

        fatigue_gain = 0.1 if success else 0.05
        self.s.fatigue = _clamp(self.s.fatigue + fatigue_gain)

        mood_change = 0.15 if success else -0.1 - 0.05 * self.s.fatigue
        self.s.mood = max(-1.0, min(1.0, self.s.mood + mood_change))

        reward_target = 1.0 if success else 0.0
        self.s.reward_history = 0.8 * self.s.reward_history + 0.2 * reward_target

        observation = self.observe()
        observation.update(
            {
                "success": success,
                "reward": reward_target,
                "success_probability": _clamp(success_p, 0.0, 1.0),
            }
        )
        return observation

    def run_brain_step(
        self,
        brain: RoboDogBrain,
        command: str,
        *,
        confidence: float = 0.85,
        reward_bias: float = 0.5,
    ) -> dict[str, Any]:
        """Execute a closed-loop interaction between the environment and brain.

        The current environment state is forwarded to the brain so that it can
        reason about the dog's mood and fatigue before selecting an action.
        Afterwards the action/score are fed back into the environment dynamics
        and the new state is returned alongside the brain's response.
        """

        current_state = self.observe()
        brain_out = brain.handle_command(
            command,
            confidence=confidence,
            reward_bias=reward_bias,
            mood=current_state["mood"],
            fatigue=current_state["fatigue"],
        )
        env_out = self.step(brain_out["action"], float(brain_out["score"]))
        return {"brain": brain_out, "state": env_out}

    def simulate_commands(
        self,
        brain: RoboDogBrain,
        commands: Sequence[str],
        *,
        confidence: float = 0.85,
        reward_bias: float = 0.5,
    ) -> dict[str, Any]:
        """Run a batch of commands through the closed-loop simulator."""

        history: list[dict[str, Any]] = []
        successes = 0
        for text in commands:
            outcome = self.run_brain_step(
                brain,
                text,
                confidence=confidence,
                reward_bias=reward_bias,
            )
            history.append(outcome)
            if outcome["state"].get("success"):
                successes += 1
        success_rate = successes / len(commands) if commands else 0.0
        return {
            "history": history,
            "success_rate": success_rate,
            "final_state": self.observe(),
        }
