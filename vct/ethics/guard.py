from dataclasses import dataclass

@dataclass
class EthicsConfig:
    max_session_min: int = 15
    min_inter_reward_s: float = 2.0
    allow_bark_reward: bool = False

class EthicsGuard:
    def __init__(self, cfg: EthicsConfig | None = None):
        self.cfg = cfg or EthicsConfig()
        self._last_reward_ts = 0.0

    def can_reward(self, now_ts: float, action: str, score: float, cooldown_s: float) -> bool:
        if action == "BARK" and not self.cfg.allow_bark_reward:
            return False
        if now_ts - self._last_reward_ts < max(cooldown_s, self.cfg.min_inter_reward_s):
            return False
        return score >= 0.6

    def note_reward(self, ts: float) -> None:
        self._last_reward_ts = ts