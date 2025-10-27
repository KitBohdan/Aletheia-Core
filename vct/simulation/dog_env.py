from dataclasses import dataclass
import random

@dataclass
class EnvState:
    fatigue: float = 0.0
    mood: float = 0.0
    reward_history: float = 0.5

class DogEnv:
    def __init__(self, seed: int = 42):
        random.seed(seed); self.s = EnvState()

    def step(self, action: str, score: float) -> dict:
        success_p = 0.5 + 0.4*score - 0.2*self.s.fatigue
        success = random.random() < max(0.05, min(0.95, success_p))
        self.s.fatigue = max(0.0, min(1.0, self.s.fatigue + (0.1 if success else 0.05)))
        self.s.mood = max(-1.0, min(1.0, self.s.mood + (0.1 if success else -0.05)))
        self.s.reward_history = (self.s.reward_history*0.8 + (1.0 if success else 0.0)*0.2)
        return {"success": success, "fatigue": self.s.fatigue, "mood": self.s.mood, "reward_hist": self.s.reward_history}