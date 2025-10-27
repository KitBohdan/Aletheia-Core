"""Abstractions around hardware reward actuators."""

from __future__ import annotations

import time


class RewardActuatorBase:
    """Base class describing a reward actuator."""

    def trigger(self, seconds: float = 0.5) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class SimulatedActuator(RewardActuatorBase):
    """Actuator that merely logs the reward trigger for simulations."""

    def trigger(self, seconds: float = 0.5) -> None:
        print(f"[REWARD] Simulated dispenser {seconds:.2f}s")
        time.sleep(min(seconds, 0.05))


class GPIOActuator(RewardActuatorBase):
    """Hardware-backed actuator driven by ``gpiozero`` if available."""

    def __init__(self, pin: int):
        try:
            from gpiozero import OutputDevice

            self.device = OutputDevice(pin)
            self.available = True
        except Exception:
            self.device = None
            self.available = False

    def trigger(self, seconds: float = 0.5) -> None:
        if not self.available or self.device is None:
            SimulatedActuator().trigger(seconds)
            return

        self.device.on()
        time.sleep(seconds)
        self.device.off()
