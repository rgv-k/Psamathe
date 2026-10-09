from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    dt: float
    duration: float

    def __post_init__(self):
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(
                "dt must be a positive finite number"
            )

        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError(
                "duration must be a positive finite number"
            )

    @property
    def n_steps(self) -> int:
        if self.duration % self.dt != 0:
            raise ValueError(
                "duration must be an exact multiple of dt"
            )

        return int(self.duration / self.dt)