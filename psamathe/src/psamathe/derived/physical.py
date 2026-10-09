import numpy as np

from psamathe.core.state import SimulationState
from psamathe.physics.gravity import G


def kinetic_energy(state: SimulationState) -> float:
    speed_squared = np.sum(state.velocity**2, axis=1)

    return float(
        0.5 * np.sum(state.mass * speed_squared)
    )


def potential_energy(state: SimulationState) -> float:
    positions = state.position
    masses = state.mass

    total = 0.0

    for i in range(state.n_bodies):
        for j in range(i + 1, state.n_bodies):

            displacement = positions[j] - positions[i]
            distance = np.linalg.norm(displacement)

            if distance == 0.0:
                raise ValueError(
                    "Potential energy is undefined for overlapping bodies"
                )

            total -= (
                G
                * masses[i]
                * masses[j]
                / distance
            )

    return float(total)


def total_energy(state: SimulationState) -> float:
    return kinetic_energy(state) + potential_energy(state)


def linear_momentum(state: SimulationState) -> np.ndarray:
    return np.sum(
        state.mass[:, np.newaxis] * state.velocity,
        axis=0,
    )


def angular_momentum(state: SimulationState) -> float:
    angular_momentum_z = np.sum(
        state.mass
        * (
            state.position[:, 0] * state.velocity[:, 1]
            - state.position[:, 1] * state.velocity[:, 0]
        )
    )

    return float(angular_momentum_z)


def pairwise_distances(state: SimulationState) -> np.ndarray:
    displacement = (
        state.position[np.newaxis, :, :]
        - state.position[:, np.newaxis, :]
    )

    distances = np.linalg.norm(
        displacement,
        axis=2,
    )

    return distances