import numpy as np

from psamathe.core.state import SimulationState
from psamathe.integrators.base import Integrator


class EulerIntegrator(Integrator):

    def step(
        self,
        state: SimulationState,
        dt: float,
        acceleration_function,
    ) -> SimulationState:

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number")

        acceleration = acceleration_function(state)

        new_position = state.position + state.velocity * dt
        new_velocity = state.velocity + acceleration * dt
        new_time = state.time + dt

        return SimulationState(
            mass=state.mass.copy(),
            position=new_position,
            velocity=new_velocity,
            time=new_time,
        )