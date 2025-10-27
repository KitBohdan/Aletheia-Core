from __future__ import annotations

import time
from typing import Any

import re

from ..behavior.policy import BehaviorInputs, BehaviorPolicy
from ..configuration import RoboDogSettings
from ..engines.stt import RuleBasedSTT, STTEngineBase, WhisperSTT
from ..engines.tts import OpenAITTS, PrintTTS, Pyttsx3TTS, TTSEngineBase
from ..ethics.guard import EthicsGuard
from ..hardware.gpio_reward import GPIOActuator, RewardActuatorBase, SimulatedActuator
from ..utils.logging import get_logger
from ..utils.metrics import record_reward

log = get_logger("RoboDogBrain")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


_NORMALIZE_PATTERN = re.compile(r"[\s_\-]+")


def _normalize_phrase(value: str) -> str:
    """Produce a comparison-friendly token from a spoken command."""

    return _NORMALIZE_PATTERN.sub("", value.casefold())


class RoboDogBrain:
    def __init__(self, cfg_path: str, gpio_pin: int | None = None, simulate: bool = False):
        self.settings = RoboDogSettings.load(cfg_path)
        self.cfg = self.settings.model_dump()
        try:
            self.stt: STTEngineBase = WhisperSTT()
        except RuntimeError as exc:
            log.warning("Whisper STT unavailable (%s), falling back to rule-based engine", exc)
            self.stt = RuleBasedSTT()
        if simulate:
            self.tts: TTSEngineBase = PrintTTS()
        elif OpenAITTS.is_configured():
            self.tts = OpenAITTS()
        else:
            self.tts = Pyttsx3TTS()
        self.policy = BehaviorPolicy(self.settings.policy_config)
        self.environment_context: dict[str, float] = self.settings.environment_context
        self.reward_map: dict[str, bool] = self.settings.reward_triggers
        self.cooldown_s = float(self.settings.reward_cooldown_s)
        self.simulate = simulate
        self.actuator: RewardActuatorBase
        if simulate or gpio_pin is None:
            self.actuator = SimulatedActuator()
        else:
            self.actuator = GPIOActuator(gpio_pin)
        self.guard = EthicsGuard()
        self._last_reward_ts = 0.0

    def _action_from_text(self, text: str) -> str:
        m = self.settings.commands_map
        normalized_input = _normalize_phrase(text)
        for phrase, action in m.items():
            if _normalize_phrase(phrase) in normalized_input:
                return action
        return "NONE"

    def _maybe_reward(self, action: str, score: float) -> bool:
        if not self.reward_map.get(action, False):
            return False
        now = time.time()
        if not self.guard.can_reward(now, action, score, self.cooldown_s):
            return False
        self.actuator.trigger(0.4)
        self.guard.note_reward(now)
        self._last_reward_ts = now
        return True

    def handle_command(
        self,
        text: str,
        confidence: float = 0.85,
        reward_bias: float = 0.5,
        mood: float = 0.0,
        fatigue: float | None = None,
    ) -> dict[str, Any]:
        action = self._action_from_text(text)
        now = time.time()
        time_since_reward = now - self._last_reward_ts if self._last_reward_ts else self.cooldown_s
        if fatigue is None:
            fatigue_value = _clamp(time_since_reward / max(self.cooldown_s * 2.0, 1.0))
        else:
            fatigue_value = _clamp(fatigue)
        stress = _clamp(1.0 - confidence)
        env_complexity = _clamp(float(self.environment_context.get("complexity", 0.5)))
        social_engagement = _clamp(float(self.environment_context.get("social_engagement", 0.5)))
        inputs = BehaviorInputs(
            stimulus=1.0 if action != "NONE" else 0.0,
            confidence=confidence,
            reward_bias=reward_bias,
            mood=mood,
            stress=stress,
            fatigue=fatigue_value,
            environmental_complexity=env_complexity,
            social_engagement=social_engagement,
        )
        vec = self.policy.decide(action, inputs)
        rewarded = self._maybe_reward(vec.action, vec.score)
        feedback = f"Дія: {vec.action} score={vec.score:.2f}" + (
            " — ✅ винагорода" if rewarded else ""
        )
        self.tts.speak(feedback)
        log.info(feedback)
        record_reward(vec.action, rewarded)
        return {"action": vec.action, "score": vec.score, "rewarded": rewarded}

    def run_once_from_wav(self, wav_path: str) -> dict[str, Any]:
        try:
            text = self.stt.transcribe(wav_path=wav_path)
        except RuntimeError as exc:
            log.warning("STT engine failed (%s); switching to rule-based fallback", exc)
            self.stt = RuleBasedSTT()
            text = self.stt.transcribe(wav_path=wav_path)
        if not text:
            self.tts.speak("Команду не розпізнано")
            return {"action": "NONE", "score": 0.0, "rewarded": False}
        return self.handle_command(text)
