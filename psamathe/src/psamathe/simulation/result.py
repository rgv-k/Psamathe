from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    mass: np.ndarray
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):

        self.mass = np.asarray(self.mass, dtype=np.float64)
        self.time = np.asarray(self.time, dtype=np.float64)
        self.position = np.asarray(
            self.position,
            dtype=np.float64,
        )
        self.velocity = np.asarray(
            self.velocity,
            dtype=np.float64,
        )

        if self.mass.ndim != 1:
            raise ValueError("mass must have shape (N,)")

        if self.time.ndim != 1:
            raise ValueError("time must have shape (steps,)")

        if self.position.ndim != 3:
            raise ValueError(
                "position must have shape (steps, N, 2)"
            )

        if self.velocity.ndim != 3:
            raise ValueError(
                "velocity must have shape (steps, N, 2)"
            )

        if self.position.shape != self.velocity.shape:
            raise ValueError(
                "position and velocity must have the same shape"
            )

        if self.position.shape[0] != self.time.shape[0]:
            raise ValueError(
                "time and state history must contain the same number of steps"
            )

        if self.position.shape[1] != self.mass.shape[0]:
            raise ValueError(
                "mass and state history must contain the same number of bodies"
            )

        if self.position.shape[2] != 2:
            raise ValueError(
                "position must have shape (steps, N, 2)"
            )

    @property
    def n_steps(self) -> int:
        return self.time.shape[0]

    @property
    def n_bodies(self) -> int:
        return self.mass.shape[0]