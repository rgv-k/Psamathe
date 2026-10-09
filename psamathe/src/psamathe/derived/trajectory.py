import numpy as np

from psamathe.core.state import SimulationState
from psamathe.derived.physical import (
    angular_momentum,
    linear_momentum,
    pairwise_distances,
    total_energy,
)
from psamathe.simulation.result import SimulationResult


def _state_at_index(
    result: SimulationResult,
    index: int,
) -> SimulationState:

    return SimulationState(
        mass=result.mass,
        position=result.position[index],
        velocity=result.velocity[index],
        time=float(result.time[index]),
    )


def total_energy_history(
    result: SimulationResult,
) -> np.ndarray:

    return np.array([
        total_energy(
            _state_at_index(result, i)
        )
        for i in range(result.n_steps)
    ])


def linear_momentum_history(
    result: SimulationResult,
) -> np.ndarray:

    return np.array([
        linear_momentum(
            _state_at_index(result, i)
        )
        for i in range(result.n_steps)
    ])


def angular_momentum_history(
    result: SimulationResult,
) -> np.ndarray:

    return np.array([
        angular_momentum(
            _state_at_index(result, i)
        )
        for i in range(result.n_steps)
    ])


def minimum_separation_history(
    result: SimulationResult,
) -> np.ndarray:

    values = []

    for i in range(result.n_steps):

        state = _state_at_index(result, i)

        distances = pairwise_distances(state)

        # Ignore diagonal entries because distance(i, i) = 0.
        nonzero_distances = distances[
            np.triu_indices(
                state.n_bodies,
                k=1,
            )
        ]

        values.append(
            np.min(nonzero_distances)
        )

    return np.array(values)


def relative_energy_error(
    result: SimulationResult,
) -> np.ndarray:

    energies = total_energy_history(result)
    initial_energy = energies[0]

    if initial_energy == 0.0:
        raise ValueError(
            "Relative energy error is undefined when "
            "initial total energy is zero"
        )

    return np.abs(
        (energies - initial_energy)
        / np.abs(initial_energy)
    )


def momentum_error(
    result: SimulationResult,
) -> np.ndarray:

    momenta = linear_momentum_history(result)
    initial_momentum = momenta[0]

    magnitude = np.linalg.norm(
        momenta - initial_momentum,
        axis=1,
    )

    reference_magnitude = np.linalg.norm(
        initial_momentum
    )

    if reference_magnitude == 0.0:

        # Initial total momentum can legitimately be zero
        # for symmetric isolated systems. Therefore we do
        # NOT divide by |P0|. Instead, the raw momentum
        # deviation is returned in SI units (kg m / s).
        return magnitude

    return magnitude / reference_magnitude


def angular_momentum_error(
    result: SimulationResult,
) -> np.ndarray:

    angular_momenta = angular_momentum_history(result)
    initial_angular_momentum = angular_momenta[0]

    deviation = np.abs(
        angular_momenta - initial_angular_momentum
    )

    reference_magnitude = abs(
        initial_angular_momentum
    )

    if reference_magnitude == 0.0:

        # Initial angular momentum can legitimately be zero.
        # Therefore we do NOT divide by |L0|. Instead, return
        # the raw angular-momentum deviation in SI units
        # (kg m^2 / s).
        return deviation

    return deviation / reference_magnitude