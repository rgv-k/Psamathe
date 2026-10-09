from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationState:
    """
    Represents the authoritative physical state of an N-body simulation.

    Each body is represented by:
        - mass
        - position
        - velocity

    The simulation time is also stored as part of the state.

    Shapes:
        mass:     (N,)
        position: (N, 2)
        velocity: (N, 2)

    where N is the number of bodies.
    """

    mass: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    time: float = 0.0

    def __post_init__(self):
        """Validate the physical state after initialization."""

        self.mass = np.asarray(self.mass, dtype=np.float64)
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)

        # Mass must be a one-dimensional array.
        if self.mass.ndim != 1:
            raise ValueError("mass must have shape (N,)")

        # Position must contain 2D coordinates.
        if self.position.ndim != 2 or self.position.shape[1] != 2:
            raise ValueError("position must have shape (N, 2)")

        # Velocity must contain 2D velocity vectors.
        if self.velocity.ndim != 2 or self.velocity.shape[1] != 2:
            raise ValueError("velocity must have shape (N, 2)")

        # All arrays must describe the same number of bodies.
        n_bodies = len(self.mass)

        if self.position.shape[0] != n_bodies:
            raise ValueError(
                "mass and position must contain the same number of bodies"
            )

        if self.velocity.shape[0] != n_bodies:
            raise ValueError(
                "mass and velocity must contain the same number of bodies"
            )

        # A simulation should contain at least three bodies.
        if n_bodies < 2:
            raise ValueError(
                "SimulationState requires at least two bodies"
            )

        # Masses must be physically meaningful.
        if np.any(self.mass <= 0):
            raise ValueError("all body masses must be positive")

        # Simulation time must be finite.
        if not np.isfinite(self.time):
            raise ValueError("simulation time must be finite")

    @property
    def n_bodies(self) -> int:
        """Return the number of bodies in the simulation."""
        return self.mass.shape[0]

    def copy(self) -> "SimulationState":
        """Return an independent copy of the simulation state."""
        return SimulationState(
            mass=self.mass.copy(),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            time=self.time,
        )