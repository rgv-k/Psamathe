
import numpy as np

from psamathe.physics.gravity import G
from psamathe.simulation.result import SimulationResult


def compute_orbital_diagnostics(
    result: SimulationResult,
) -> dict[str, float | bool]:
    """
    Calculate classical two-body orbital diagnostics.

    These calculations assume two bodies interacting through
    Newtonian gravity. They describe the relative orbit of body 2
    with respect to body 1.

    The orbital period and eccentricity are derived from the
    initial relative position and velocity.
    """

    if result.n_bodies != 2:
        raise ValueError(
            "Orbital diagnostics require exactly two bodies."
        )

    masses = np.asarray(result.mass, dtype=float)
    positions = np.asarray(result.position, dtype=float)
    velocities = np.asarray(result.velocity, dtype=float)
    times = np.asarray(result.time, dtype=float)

    if len(times) < 2:
        raise ValueError(
            "At least two recorded states are required."
        )

    if not (
        np.all(np.isfinite(masses))
        and np.all(np.isfinite(positions))
        and np.all(np.isfinite(velocities))
        and np.all(np.isfinite(times))
    ):
        raise ValueError(
            "Orbital diagnostics require finite recorded values."
        )

    # Relative state: body 2 with respect to body 1.
    r0 = positions[0, 1] - positions[0, 0]
    v0 = velocities[0, 1] - velocities[0, 0]

    r0_norm = np.linalg.norm(r0)

    if r0_norm == 0:
        raise ValueError(
            "Initial body separation must be nonzero."
        )

    mu = G * np.sum(masses)

    # Specific relative orbital energy.
    specific_energy = (
        0.5 * np.dot(v0, v0)
        - mu / r0_norm
    )

    # Angular momentum per unit reduced mass, in the z direction.
    h_z = r0[0] * v0[1] - r0[1] * v0[0]

    # Eccentricity vector for a Newtonian two-body orbit.
    eccentricity_vector = np.array([
        v0[1] * h_z / mu - r0[0] / r0_norm,
        -v0[0] * h_z / mu - r0[1] / r0_norm,
    ])

    eccentricity = float(
        np.linalg.norm(eccentricity_vector)
    )

    bound_orbit = specific_energy < 0

    if bound_orbit:
        semi_major_axis = -mu / (2.0 * specific_energy)

        orbital_period = float(
            2.0 * np.pi
            * np.sqrt(semi_major_axis**3 / mu)
        )
    else:
        semi_major_axis = float("nan")
        orbital_period = float("nan")

    # Centre of mass throughout the recorded trajectory.
    total_mass = np.sum(masses)

    center_of_mass = np.sum(
        positions * masses[np.newaxis, :, np.newaxis],
        axis=1,
    ) / total_mass

    center_of_mass_drift = np.linalg.norm(
        center_of_mass - center_of_mass[0],
        axis=1,
    )

    # Relative orbital position at the final recorded state.
    final_relative_position = (
        positions[-1, 1] - positions[-1, 0]
    )

    final_position_difference = float(
        np.linalg.norm(final_relative_position - r0)
    )

    final_position_difference_fraction = (
        final_position_difference / r0_norm
    )

    elapsed_time = float(times[-1] - times[0])

    if np.isfinite(orbital_period):
        period_fraction = elapsed_time / orbital_period
    else:
        period_fraction = float("nan")

    return {
        "eccentricity": eccentricity,
        "semi_major_axis_m": float(semi_major_axis),
        "orbital_period_s": orbital_period,
        "bound_orbit": bool(bound_orbit),
        "center_of_mass_drift_m": float(
            np.max(center_of_mass_drift)
        ),
        "initial_separation_m": float(r0_norm),
        "final_position_difference_m": final_position_difference,
        "final_position_difference_fraction": float(
            final_position_difference_fraction
        ),
        "elapsed_time_s": elapsed_time,
        "elapsed_periods": float(period_fraction),
    }
