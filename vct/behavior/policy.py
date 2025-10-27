from dataclasses import dataclass
from typing import Dict

@dataclass
class BehaviorInputs:
    stimulus: float
    confidence: float
    reward_bias: float
    mood: float  # -1..1

@dataclass
class BehaviorVector:
    score: float
    action: str

class BehaviorPolicy:
    def __init__(self, weights: Dict[str, float]):
        self.w = weights

    def decide(self, action: str, inputs: BehaviorInputs) -> BehaviorVector:
        score = (
            self.w.get("stimulus", 0.4) * inputs.stimulus +
            self.w.get("confidence", 0.3) * inputs.confidence +
            self.w.get("reward_bias", 0.2) * inputs.reward_bias +
            self.w.get("mood", 0.1) * ((inputs.mood + 1) / 2.0)
        )
        # clamp 0..1
        score = max(0.0, min(1.0, score))
        return BehaviorVector(score=score, action=action)