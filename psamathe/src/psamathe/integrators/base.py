from abc import ABC, abstractmethod

from psamathe.core.state import SimulationState


class Integrator(ABC):
    @abstractmethod
    def step(
        self,
        state: SimulationState,
        dt: float,
        acceleration_function,
    ) -> SimulationState:
        """
        Advance the simulation by one timestep.

        Parameters
        ----------
        state : SimulationState
            Current simulation state.

        dt : float
            Integration timestep in seconds.

        acceleration_function : callable
            Function that computes accelerations from the current state.

        Returns
        -------
        SimulationState
            New state after advancing by dt.
        """
        pass