from typing import Callable

import numpy as np

from psamathe.core.state import SimulationState
from psamathe.integrators.base import Integrator
from psamathe.simulation.result import SimulationResult

class SimulationEngine:

    def __init__(
        self,
        state: SimulationState,
        integrator: Integrator,
        acceleration_function: Callable,
        dt: float,
    ):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number")

        self.state = state.copy()
        self.integrator = integrator
        self.acceleration_function = acceleration_function
        self.dt = dt

    def step(self) -> SimulationState:
        self.state = self.integrator.step(
            self.state,
            self.dt,
            self.acceleration_function,
        )

        return self.state

    def run(self, n_steps: int, record_history: bool = False):
        if not isinstance(n_steps, int):
            raise TypeError("n_steps must be an integer")

        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        if not record_history:
            for _ in range(n_steps):
                self.step()

            return self.state

        from psamathe.simulation.recorder import SimulationRecorder

        recorder = SimulationRecorder(self.state)

        for _ in range(n_steps):
            self.step()
            recorder.record(self.state)

        return recorder.result()