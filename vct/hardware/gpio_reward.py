import time
class RewardActuatorBase:
    def trigger(self, seconds: float = 0.5) -> None: raise NotImplementedError

class SimulatedActuator(RewardActuatorBase):
    def trigger(self, seconds: float = 0.5) -> None:
        print(f"[REWARD] Simulated dispenser {seconds:.2f}s"); time.sleep(min(seconds, 0.05))

class GPIOActuator(RewardActuatorBase):
    def __init__(self, pin: int):
        try:
            from gpiozero import OutputDevice  # type: ignore
            self.device = OutputDevice(pin); self.available = True
        except Exception:
            self.device = None; self.available = False
    def trigger(self, seconds: float = 0.5) -> None:
        if not self.available or self.device is None:
            SimulatedActuator().trigger(seconds); return
        self.device.on(); time.sleep(seconds); self.device.off()