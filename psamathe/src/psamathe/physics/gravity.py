import numpy as np

from psamathe.core.state import SimulationState


# SI units:
# m^3 kg^-1 s^-2
G = 6.67430e-11


def compute_accelerations(state: SimulationState) -> np.ndarray:
    """
    Compute the gravitational acceleration of every body.

    Parameters
    ----------
    state : SimulationState
        Current N-body physical state.

    Returns
    -------
    np.ndarray
        Accelerations with shape (N, 2), in m/s^2.

    Notes
    -----
    The calculation uses Newtonian point-mass gravity:

        a_i = G * sum_j != i [
            m_j * (r_j - r_i) / |r_j - r_i|^3
        ]

    No gravitational softening is applied.
    """

    positions = state.position
    masses = state.mass

    # Pairwise displacement:
    #
    # displacement[i, j] = position[j] - position[i]
    #
    # Shape:
    # (N, N, 2)
    displacement = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

    # Pairwise squared distances.
    #
    # Shape:
    # (N, N)
    distance_squared = np.sum(displacement**2, axis=2)

    # We never want a body to exert gravity on itself.
    #
    # Set the diagonal to infinity so that its contribution becomes zero.
    np.fill_diagonal(distance_squared, np.inf)

    # |r_ij|^3
    distance_cubed = distance_squared ** 1.5

    # Gravitational acceleration contribution from every body j
    # acting on every body i.
    #
    # masses[np.newaxis, :, np.newaxis] gives shape (1, N, 1)
    # and broadcasts over the receiving body i.
    acceleration_contributions = (
        G
        * masses[np.newaxis, :, np.newaxis]
        * displacement
        / distance_cubed[:, :, np.newaxis]
    )

    # Sum contributions from every other body.
    accelerations = np.sum(acceleration_contributions, axis=1)

    return accelerations