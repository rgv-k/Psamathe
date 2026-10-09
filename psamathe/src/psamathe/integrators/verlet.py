import numpy as np

from psamathe.core.state import SimulationState
from psamathe.integrators.base import Integrator


class VelocityVerletIntegrator(Integrator):

    def step(
        self,
        state: SimulationState,
        dt: float,
        acceleration_function,
    ) -> SimulationState:

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number")

        # Acceleration at current state
        acceleration = acceleration_function(state)

        # Update position
        new_position = (
            state.position
            + state.velocity * dt
            + 0.5 * acceleration * dt**2
        )

        # Create temporary state at the new position
        intermediate_state = SimulationState(
            mass=state.mass.copy(),
            position=new_position,
            velocity=state.velocity.copy(),
            time=state.time + dt,
        )

        # Acceleration at new position
        new_acceleration = acceleration_function(intermediate_state)

        # Update velocity
        new_velocity = (
            state.velocity
            + 0.5 * (acceleration + new_acceleration) * dt
        )

        return SimulationState(
            mass=state.mass.copy(),
            position=new_position,
            velocity=new_velocity,
            time=state.time + dt,
        )