from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class BehaviorInputs:
    stimulus: float
    confidence: float
    reward_bias: float
    mood: float  # -1..1
    stress: float = 0.0
    fatigue: float = 0.0
    environmental_complexity: float = 0.0
    social_engagement: float = 0.0

    def as_vector(self) -> List[float]:
        """Return a normalized feature vector ready for the policy network."""
        mood_normalized = _clamp((self.mood + 1.0) / 2.0)
        return [
            _clamp(self.stimulus),
            _clamp(self.confidence),
            _clamp(self.reward_bias),
            mood_normalized,
            _clamp(self.stress),
            _clamp(self.fatigue),
            _clamp(self.environmental_complexity),
            _clamp(self.social_engagement),
        ]

    @classmethod
    def feature_names(cls) -> Tuple[str, ...]:
        return (
            "stimulus",
            "confidence",
            "reward_bias",
            "mood",
            "stress",
            "fatigue",
            "environmental_complexity",
            "social_engagement",
        )


@dataclass
class BehaviorVector:
    score: float
    action: str


_DEFAULT_WEIGHT_MAP: Dict[str, float] = {
    "stimulus": 0.4,
    "confidence": 0.3,
    "reward_bias": 0.2,
    "mood": 0.1,
    "stress": -0.1,
    "fatigue": -0.1,
    "environmental_complexity": -0.05,
    "social_engagement": 0.05,
}


class _AdaptiveMLP:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        learning_rate: float,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self._rng = rng or random.Random()
        scale = 1.0 / math.sqrt(max(1, input_size))
        self.hidden_weights = [
            [self._rng.uniform(-scale, scale) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.hidden_bias = [0.0 for _ in range(hidden_size)]
        self.output_weights = [self._rng.uniform(-scale, scale) for _ in range(hidden_size)]
        self.output_bias = 0.0

    def forward(self, features: Sequence[float]) -> Tuple[float, List[float]]:
        hidden: List[float] = []
        for neuron_weights, bias in zip(self.hidden_weights, self.hidden_bias):
            activation = sum(w * x for w, x in zip(neuron_weights, features)) + bias
            hidden.append(math.tanh(activation))
        output_activation = sum(w * h for w, h in zip(self.output_weights, hidden)) + self.output_bias
        score = 1.0 / (1.0 + math.exp(-output_activation))
        return score, hidden

    def predict(self, features: Sequence[float]) -> float:
        score, _ = self.forward(features)
        return score

    def train_step(self, features: Sequence[float], target: float) -> float:
        score, hidden = self.forward(features)
        error = score - target
        d_output = error * score * (1.0 - score)

        # Output layer gradients
        grad_output_weights = [d_output * h for h in hidden]
        grad_output_bias = d_output

        # Hidden layer gradients
        grad_hidden: List[float] = []
        for weight, h_val in zip(self.output_weights, hidden):
            grad_hidden.append((1.0 - h_val ** 2) * weight * d_output)

        # Update output layer
        for i in range(self.hidden_size):
            self.output_weights[i] -= self.learning_rate * grad_output_weights[i]
        self.output_bias -= self.learning_rate * grad_output_bias

        # Update hidden layer
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.hidden_weights[i][j] -= self.learning_rate * grad_hidden[i] * features[j]
            self.hidden_bias[i] -= self.learning_rate * grad_hidden[i]

        return 0.5 * (score - target) ** 2

    def set_linear_mapping(self, feature_weights: Sequence[float], bias: float = 0.0) -> None:
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.hidden_weights[i][j] = 0.0
            if i < self.input_size:
                self.hidden_weights[i][i] = 1.0
            self.hidden_bias[i] = 0.0
        for i in range(self.hidden_size):
            self.output_weights[i] = feature_weights[i] if i < len(feature_weights) else 0.0
        self.output_bias = bias


TrainingExample = Tuple[BehaviorInputs, float]


class BehaviorPolicy:
    def __init__(self, config: Optional[Dict] = None):
        config = dict(config or {})
        legacy_weights: Dict[str, float] = {}
        if config and all(isinstance(v, (int, float)) for v in config.values()):
            legacy_weights = {str(k): float(v) for k, v in config.items()}
            config = {}
        else:
            raw_weights = config.get("weights", {})
            if isinstance(raw_weights, dict):
                legacy_weights = {str(k): float(v) for k, v in raw_weights.items()}
        learning_rate = float(config.get("learning_rate", 0.05))
        feature_count = len(BehaviorInputs.feature_names())
        hidden_size = max(int(config.get("hidden_size", feature_count)), feature_count)
        seed = config.get("seed")
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._model = _AdaptiveMLP(
            input_size=feature_count,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            rng=self._rng,
        )
        self._trained = False
        self.training_history: List[float] = []
        if legacy_weights:
            self._warm_start_from_weights(legacy_weights)
        training_data = config.get("training_data", [])
        if training_data:
            dataset = list(self._parse_training_data(training_data))
            if dataset:
                epochs = int(config.get("epochs", 150))
                self.train(dataset, epochs=epochs)

    def _warm_start_from_weights(self, weight_map: Dict[str, float]) -> None:
        features = []
        for name in BehaviorInputs.feature_names():
            default = _DEFAULT_WEIGHT_MAP.get(name, 0.0)
            features.append(weight_map.get(name, default))
        bias = weight_map.get("bias", 0.0)
        self._model.set_linear_mapping(features, bias)

    def _parse_training_data(self, training_data: Iterable) -> Iterable[TrainingExample]:
        for item in training_data:
            if isinstance(item, tuple) and len(item) == 2:
                inputs, target = item
                if isinstance(inputs, BehaviorInputs):
                    yield inputs, _clamp(float(target))
            elif isinstance(item, dict):
                inputs_payload = item.get("inputs")
                if isinstance(inputs_payload, dict) and "target" in item:
                    inputs = BehaviorInputs(**inputs_payload)
                    yield inputs, _clamp(float(item["target"]))
                elif isinstance(inputs_payload, dict) and "score" in item:
                    inputs = BehaviorInputs(**inputs_payload)
                    yield inputs, _clamp(float(item["score"]))

    def train(self, dataset: Sequence[TrainingExample], epochs: int = 150) -> List[float]:
        if not dataset:
            return []
        data = list(dataset)
        history: List[float] = []
        for _ in range(max(1, epochs)):
            self._rng.shuffle(data)
            total_loss = 0.0
            for inputs, target in data:
                features = inputs.as_vector()
                total_loss += self._model.train_step(features, _clamp(target))
            history.append(total_loss / len(data))
        self.training_history.extend(history)
        self._trained = True
        return history

    def decide(self, action: str, inputs: BehaviorInputs) -> BehaviorVector:
        score = self._model.predict(inputs.as_vector())
        score = _clamp(score)
        return BehaviorVector(score=score, action=action)
