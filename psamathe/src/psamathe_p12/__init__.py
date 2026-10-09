"""Psamathe P1.2 - Graphical and Computational Representation layer.

Importing this package does NOT import matplotlib; the viewer is loaded only
from ``psamathe_p12.viewer`` so the data model and representation logic stay
usable (and testable) without any GUI stack.
"""

from .adapters import CallableStateProvider, IterableStateProvider, coerce_state
from .config import RepresentationConfig
from .provider import SimulationStateProvider
from .representation import Indicator, StateRepresentation, VectorGlyph
from .state import (
    ParticleState,
    SimulationState,
    StateValidationError,
    particle_from_mapping,
    state_from_mapping,
)

__all__ = [
    "CallableStateProvider", "IterableStateProvider", "coerce_state",
    "RepresentationConfig", "SimulationStateProvider",
    "Indicator", "StateRepresentation", "VectorGlyph",
    "ParticleState", "SimulationState", "StateValidationError",
    "particle_from_mapping", "state_from_mapping",
]
__version__ = "0.1.0"
