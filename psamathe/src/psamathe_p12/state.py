"""P1.2 state interface: the snapshot data model.

These immutable dataclasses are the *only* thing P1.2 knows about the
simulation.  A future P1.1 simulator (or an adapter around it) must produce
``SimulationState`` objects; P1.2 never imports or inspects P1.1 internals.

Design rules
------------
* Snapshots are immutable (``frozen=True``): the representation layer cannot
  modify the physical state it is shown.
* Everything is validated on construction, so a malformed snapshot fails
  loudly at the boundary (``StateValidationError``) instead of corrupting a
  display later.
* Optional quantities default to ``None`` and are shown as "unavailable".
  Nothing is ever estimated or fabricated by P1.2.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

ParticleId = Union[int, str]
ExtraValue = Union[int, float, str]


class StateValidationError(ValueError):
    """Raised when a snapshot (or part of one) violates the state schema."""


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------
def _as_float(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    allow_none: bool = False,
) -> Optional[float]:
    """Coerce ``value`` to a finite ``float`` or raise ``StateValidationError``."""
    if value is None:
        if allow_none:
            return None
        raise StateValidationError(f"{name} is required but was None")
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise StateValidationError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise StateValidationError(f"{name} must be finite, got {result}")
    if positive and result <= 0.0:
        raise StateValidationError(f"{name} must be > 0, got {result}")
    return result


def _as_optional_str(name: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise StateValidationError(f"{name} must be a string, got {type(value).__name__}")
    return value


# --------------------------------------------------------------------------
# ParticleState
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ParticleState:
    """State of a single particle at one instant (2D).

    ``force_x`` / ``force_y`` are optional but must be given together.
    """

    id: ParticleId
    x: float
    y: float
    vx: float
    vy: float
    mass: float
    force_x: Optional[float] = None
    force_y: Optional[float] = None

    def __post_init__(self) -> None:
        pid = self.id
        if isinstance(pid, bool):
            raise StateValidationError("particle id must be int or str, not bool")
        if isinstance(pid, numbers.Integral):
            object.__setattr__(self, "id", int(pid))
        elif isinstance(pid, str):
            if not pid.strip():
                raise StateValidationError("particle id must not be an empty string")
        else:
            raise StateValidationError(
                f"particle id must be int or str, got {type(pid).__name__}"
            )

        for name in ("x", "y", "vx", "vy"):
            object.__setattr__(self, name, _as_float(f"particle {pid!r}: {name}", getattr(self, name)))
        object.__setattr__(
            self, "mass", _as_float(f"particle {pid!r}: mass", self.mass, positive=True)
        )

        fx = _as_float(f"particle {pid!r}: force_x", self.force_x, allow_none=True)
        fy = _as_float(f"particle {pid!r}: force_y", self.force_y, allow_none=True)
        if (fx is None) != (fy is None):
            raise StateValidationError(
                f"particle {pid!r}: force_x and force_y must be provided together"
            )
        object.__setattr__(self, "force_x", fx)
        object.__setattr__(self, "force_y", fy)

    @property
    def has_force(self) -> bool:
        return self.force_x is not None

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(id=self.id, x=self.x, y=self.y, vx=self.vx, vy=self.vy, mass=self.mass)
        if self.has_force:
            data.update(force_x=self.force_x, force_y=self.force_y)
        return data


# --------------------------------------------------------------------------
# SimulationState
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationState:
    """One immutable snapshot of the simulation, as seen by P1.2.

    Required: ``simulation_time``, ``timestep`` (may be ``None`` if the
    source cannot report it) and ``particles`` (may be empty).

    Everything else is optional numerical/computational metadata that is
    displayed only if the source provides it.
    """

    simulation_time: float
    timestep: Optional[float]
    particles: Sequence[ParticleState]
    step: Optional[int] = None
    integration_method: Optional[str] = None
    numerical_precision: Optional[str] = None
    total_energy: Optional[float] = None
    energy_error: Optional[float] = None
    extras: Mapping[str, ExtraValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "simulation_time", _as_float("simulation_time", self.simulation_time)
        )
        object.__setattr__(
            self, "timestep", _as_float("timestep", self.timestep, positive=True, allow_none=True)
        )

        if isinstance(self.particles, (str, bytes)) or not hasattr(self.particles, "__iter__"):
            raise StateValidationError("particles must be a sequence of ParticleState")
        particles: Tuple[ParticleState, ...] = tuple(self.particles)
        seen = set()
        for p in particles:
            if not isinstance(p, ParticleState):
                raise StateValidationError(
                    f"particles must contain ParticleState objects, got {type(p).__name__}"
                )
            if p.id in seen:
                raise StateValidationError(f"duplicate particle id {p.id!r}")
            seen.add(p.id)
        object.__setattr__(self, "particles", particles)

        if self.step is not None:
            if isinstance(self.step, bool) or not isinstance(self.step, numbers.Integral) or self.step < 0:
                raise StateValidationError(f"step must be a non-negative integer, got {self.step!r}")
            object.__setattr__(self, "step", int(self.step))

        object.__setattr__(self, "integration_method", _as_optional_str("integration_method", self.integration_method))
        object.__setattr__(self, "numerical_precision", _as_optional_str("numerical_precision", self.numerical_precision))
        object.__setattr__(self, "total_energy", _as_float("total_energy", self.total_energy, allow_none=True))
        object.__setattr__(self, "energy_error", _as_float("energy_error", self.energy_error, allow_none=True))

        if not isinstance(self.extras, Mapping):
            raise StateValidationError("extras must be a mapping")
        clean: Dict[str, ExtraValue] = {}
        for key, value in self.extras.items():
            if not isinstance(key, str):
                raise StateValidationError(f"extras keys must be str, got {type(key).__name__}")
            if isinstance(value, bool) or not isinstance(value, (numbers.Real, str)):
                raise StateValidationError(f"extras[{key!r}] must be a number or string")
            clean[key] = value if isinstance(value, str) else float(value)
        object.__setattr__(self, "extras", clean)

    @property
    def particle_count(self) -> int:
        return len(self.particles)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_time": self.simulation_time,
            "timestep": self.timestep,
            "step": self.step,
            "integration_method": self.integration_method,
            "numerical_precision": self.numerical_precision,
            "total_energy": self.total_energy,
            "energy_error": self.energy_error,
            "extras": dict(self.extras),
            "particles": [p.to_dict() for p in self.particles],
        }


# --------------------------------------------------------------------------
# Plain-dict conversion (convenient for adapters / JSON replay)
# --------------------------------------------------------------------------
_PARTICLE_REQUIRED = ("id", "x", "y", "vx", "vy", "mass")


def particle_from_mapping(data: Mapping[str, Any]) -> ParticleState:
    """Build a ``ParticleState`` from a plain mapping."""
    if not isinstance(data, Mapping):
        raise StateValidationError(f"particle must be a mapping, got {type(data).__name__}")
    missing = [k for k in _PARTICLE_REQUIRED if k not in data]
    if missing:
        raise StateValidationError(f"particle is missing required field(s): {missing}")
    return ParticleState(
        id=data["id"], x=data["x"], y=data["y"], vx=data["vx"], vy=data["vy"],
        mass=data["mass"], force_x=data.get("force_x"), force_y=data.get("force_y"),
    )


def state_from_mapping(data: Mapping[str, Any]) -> SimulationState:
    """Build a ``SimulationState`` from a plain mapping.

    Required keys: ``simulation_time``, ``particles``.  All other schema keys
    are optional.  Unknown keys are ignored (put extra scalars in ``extras``).
    """
    if not isinstance(data, Mapping):
        raise StateValidationError(f"snapshot must be a mapping, got {type(data).__name__}")
    for key in ("simulation_time", "particles"):
        if key not in data:
            raise StateValidationError(f"snapshot is missing required field {key!r}")
    raw_particles = data["particles"]
    if isinstance(raw_particles, (str, bytes, Mapping)) or not hasattr(raw_particles, "__iter__"):
        raise StateValidationError("'particles' must be a list of particle mappings")
    particles = []
    for index, item in enumerate(raw_particles):
        try:
            particles.append(particle_from_mapping(item))
        except StateValidationError as exc:
            raise StateValidationError(f"particle at index {index}: {exc}") from exc
    return SimulationState(
        simulation_time=data["simulation_time"],
        timestep=data.get("timestep"),
        particles=particles,
        step=data.get("step"),
        integration_method=data.get("integration_method"),
        numerical_precision=data.get("numerical_precision"),
        total_energy=data.get("total_energy"),
        energy_error=data.get("energy_error"),
        extras=data.get("extras") or {},
    )
