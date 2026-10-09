from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentConfiguration:
    """Configuration captured before an experiment starts."""

    name: str
    scenario: str
    n_bodies: int
    duration: float
    integrator: str
    dt: float
    bodies: list[dict[str, float]]

    @property
    def precision(self) -> str:
        return "float64"


@dataclass
class ExperimentSession:
    """GUI-level session state.

    This is deliberately separate from SimulationState. The GUI should
    not become the owner of the simulator's physical state.
    """

    configuration: ExperimentConfiguration | None = None
    result: Any = None
