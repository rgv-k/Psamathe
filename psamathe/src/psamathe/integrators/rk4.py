import numpy as np

from psamathe.core.state import SimulationState
from psamathe.integrators.base import Integrator


class RK4Integrator(Integrator):

    def step(
        self,
        state: SimulationState,
        dt: float,
        acceleration_function,
    ) -> SimulationState:

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number")

        # k1
        a1 = acceleration_function(state)
        v1 = state.velocity

        # k2
        position_2 = state.position + 0.5 * v1 * dt
        velocity_2 = state.velocity + 0.5 * a1 * dt

        state_2 = SimulationState(
            mass=state.mass.copy(),
            position=position_2,
            velocity=velocity_2,
            time=state.time + 0.5 * dt,
        )

        a2 = acceleration_function(state_2)
        v2 = velocity_2

        # k3
        position_3 = state.position + 0.5 * v2 * dt
        velocity_3 = state.velocity + 0.5 * a2 * dt

        state_3 = SimulationState(
            mass=state.mass.copy(),
            position=position_3,
            velocity=velocity_3,
            time=state.time + 0.5 * dt,
        )

        a3 = acceleration_function(state_3)
        v3 = velocity_3

        # k4
        position_4 = state.position + v3 * dt
        velocity_4 = state.velocity + a3 * dt

        state_4 = SimulationState(
            mass=state.mass.copy(),
            position=position_4,
            velocity=velocity_4,
            time=state.time + dt,
        )

        a4 = acceleration_function(state_4)
        v4 = velocity_4

        # Combine slopes
        new_position = state.position + (
            dt / 6.0
        ) * (
            v1 + 2.0 * v2 + 2.0 * v3 + v4
        )

        new_velocity = state.velocity + (
            dt / 6.0
        ) * (
            a1 + 2.0 * a2 + 2.0 * a3 + a4
        )

        return SimulationState(
            mass=state.mass.copy(),
            position=new_position,
            velocity=new_velocity,
            time=state.time + dt,
        )