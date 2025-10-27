from vct.hardware.gpio_reward import SimulatedActuator, GPIOActuator

def test_simulated_actuator_runs():
    SimulatedActuator().trigger(0.01)

def test_gpio_actuator_fallback():
    g = GPIOActuator(pin=18)
    g.trigger(0.01)