import numpy as np

from psamathe.core.state import SimulationState
from psamathe.simulation.result import SimulationResult


class SimulationRecorder:
    """
    Records simulation states and converts them into a SimulationResult.
    """

    def __init__(self, initial_state: SimulationState):
        self._mass = initial_state.mass.copy()

        self._time = [
            initial_state.time
        ]

        self._position = [
            initial_state.position.copy()
        ]

        self._velocity = [
            initial_state.velocity.copy()
        ]

    def record(self, state: SimulationState) -> None:
        """
        Record a copy of the current simulation state.
        """

        self._time.append(state.time)
        self._position.append(state.position.copy())
        self._velocity.append(state.velocity.copy())

    def result(self) -> SimulationResult:
        """
        Build a SimulationResult from all recorded states.
        """

        return SimulationResult(
            mass=self._mass.copy(),
            time=np.asarray(self._time, dtype=np.float64),
            position=np.asarray(self._position, dtype=np.float64),
            velocity=np.asarray(self._velocity, dtype=np.float64),
        )

    @property
    def position_history(self):
        """Return the recorded position history."""

        return np.asarray(
            self._position,
            dtype=np.float64,
        )