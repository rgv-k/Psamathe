"""P1.2 representation layer (GUI-independent).

``StateRepresentation`` consumes ``SimulationState`` snapshots and maintains
exactly what a display needs: the latest state, bounded trajectory histories,
scaled vector glyphs, axis bounds and the list of numerical indicators.

It is a pure *consumer*:
* it never modifies a snapshot (they are frozen),
* it never talks back to the source (only ``next_snapshot()`` is called),
* it performs no physics and derives no scientific quantities.

Rendering lives in ``viewer.py``; this module needs no plotting library.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from .config import RepresentationConfig
from .provider import SimulationStateProvider
from .state import ParticleId, SimulationState, StateValidationError

UNAVAILABLE = "unavailable"
Bounds = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


@dataclass(frozen=True)
class Indicator:
    """One labelled numerical/computational indicator.  ``value`` is ``None`` if unavailable."""

    label: str
    value: Optional[str]

    @property
    def available(self) -> bool:
        return self.value is not None

    @property
    def display(self) -> str:
        return self.value if self.value is not None else UNAVAILABLE


@dataclass(frozen=True)
class VectorGlyph:
    """An arrow anchored at (x, y) with display-scaled components (dx, dy)."""

    id: ParticleId
    x: float
    y: float
    dx: float
    dy: float
    magnitude: float  # unscaled magnitude of the underlying quantity


def _fmt(value: Optional[float], spec: str = ".6g") -> Optional[str]:
    return None if value is None else format(value, spec)


class StateRepresentation:
    """Stateful consumer of snapshots; the data model behind the visualization."""

    def __init__(self, config: Optional[RepresentationConfig] = None) -> None:
        self.config = config or RepresentationConfig()
        self._state: Optional[SimulationState] = None
        self._trajectories: Dict[ParticleId, Deque[Tuple[float, float]]] = {}
        self._running_bounds: Optional[List[float]] = None
        self._max_speed = 0.0
        self._max_force = 0.0
        # Bookkeeping of the representation layer itself (not simulation data)
        self.snapshots_received = 0
        self.snapshots_rejected = 0
        self.history_resets = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------ input
    @property
    def state(self) -> Optional[SimulationState]:
        """The most recent valid snapshot (``None`` before the first one)."""
        return self._state

    def update(self, state: SimulationState) -> None:
        """Accept a snapshot.  Raises ``StateValidationError`` if it is not valid."""
        if not isinstance(state, SimulationState):
            raise StateValidationError(
                f"expected SimulationState, got {type(state).__name__}"
            )
        previous = self._state
        # Simulation time going backwards means a restart: old history is meaningless.
        if previous is not None and state.simulation_time < previous.simulation_time:
            self._reset_history()
            previous = None
        new_instant = previous is None or state.simulation_time != previous.simulation_time

        current_ids = {p.id for p in state.particles}
        for stale in set(self._trajectories) - current_ids:
            del self._trajectories[stale]  # particle removed -> free its history

        limit = self.config.trajectory_history_length
        for p in state.particles:
            history = self._trajectories.get(p.id)
            if history is None:
                history = self._trajectories[p.id] = deque(maxlen=limit)
                history.append((p.x, p.y))
            elif new_instant:
                history.append((p.x, p.y))
            self._extend_bounds(p.x, p.y)
            self._max_speed = max(self._max_speed, p.speed)
            if p.has_force:
                self._max_force = max(self._max_force, abs(p.force_x) + abs(p.force_y))  # type: ignore[arg-type]

        self._state = state
        self.snapshots_received += 1

    def try_update(self, state: object) -> bool:
        """Like ``update`` but records the error and returns ``False`` instead of raising."""
        try:
            self.update(state)  # type: ignore[arg-type]
        except StateValidationError as exc:
            self._reject(exc)
            return False
        return True

    def poll(self, provider: SimulationStateProvider) -> bool:
        """Pull one snapshot from ``provider``.  Returns ``True`` if the display state changed.

        A malformed snapshot is rejected and counted; the last good snapshot
        stays on screen.  Other exceptions (e.g. a crashing simulator) are
        deliberately *not* swallowed.
        """
        try:
            snapshot = provider.next_snapshot()
        except StateValidationError as exc:
            self._reject(exc)
            return False
        if snapshot is None:
            return False
        return self.try_update(snapshot)

    def _reject(self, exc: Exception) -> None:
        self.snapshots_rejected += 1
        self.last_error = str(exc)

    def _reset_history(self) -> None:
        self._trajectories.clear()
        self._running_bounds = None
        self._max_speed = 0.0
        self._max_force = 0.0
        self.history_resets += 1

    # ------------------------------------------------------------ trajectories
    def trajectory(self, particle_id: ParticleId) -> List[Tuple[float, float]]:
        """Recent positions of one particle, oldest first (empty if unknown)."""
        return list(self._trajectories.get(particle_id, ()))

    def trajectory_point_count(self) -> int:
        """Total stored trajectory points (useful to verify the memory bound)."""
        return sum(len(d) for d in self._trajectories.values())

    # ----------------------------------------------------------------- bounds
    def _extend_bounds(self, x: float, y: float) -> None:
        if self._running_bounds is None:
            self._running_bounds = [x, x, y, y]
        else:
            b = self._running_bounds
            b[0], b[1] = min(b[0], x), max(b[1], x)
            b[2], b[3] = min(b[2], y), max(b[3], y)

    def view_bounds(self) -> Bounds:
        """Axis limits (xmin, xmax, ymin, ymax), square so that aspect is 1:1."""
        cfg = self.config
        if cfg.axis_mode == "fixed":
            assert cfg.fixed_extent is not None
            return cfg.fixed_extent

        raw: Optional[Bounds] = None
        if self._state is not None and self._state.particles:
            if cfg.axis_mode == "expand" and self._running_bounds is not None:
                b = self._running_bounds
                raw = (b[0], b[1], b[2], b[3])
            else:  # "fit": current positions plus the retained trajectory points
                xs = [p.x for p in self._state.particles]
                ys = [p.y for p in self._state.particles]
                for hist in self._trajectories.values():
                    xs.extend(pt[0] for pt in hist)
                    ys.extend(pt[1] for pt in hist)
                raw = (min(xs), max(xs), min(ys), max(ys))
        if raw is None:
            return (-1.0, 1.0, -1.0, 1.0)

        cx, cy = (raw[0] + raw[1]) / 2, (raw[2] + raw[3]) / 2
        half = max(raw[1] - raw[0], raw[3] - raw[2]) / 2
        half = (half * (1 + 2 * cfg.axis_margin)) if half > 0 else 1.0
        return (cx - half, cx + half, cy - half, cy + half)

    # ---------------------------------------------------------------- vectors
    def _effective_scale(self, explicit: Optional[float], running_max: float) -> float:
        if explicit is not None:
            return explicit
        if running_max <= 0:
            return 1.0
        xmin, xmax, _, _ = self.view_bounds()
        return self.config.auto_vector_fraction * (xmax - xmin) / running_max

    def velocity_scale_in_use(self) -> float:
        """Display length per unit velocity currently applied to velocity arrows."""
        return self._effective_scale(self.config.velocity_scale, self._max_speed)

    def velocity_vectors(self) -> List[VectorGlyph]:
        """Velocity arrows scaled so they stay readable (see ``RepresentationConfig``)."""
        if self._state is None:
            return []
        k = self.velocity_scale_in_use()
        return [VectorGlyph(p.id, p.x, p.y, p.vx * k, p.vy * k, p.speed) for p in self._state.particles]

    def force_vectors(self) -> List[VectorGlyph]:
        """Force arrows for particles that provide forces (never estimated)."""
        if self._state is None:
            return []
        k = self._effective_scale(self.config.force_scale, self._max_force)
        glyphs = []
        for p in self._state.particles:
            if p.has_force:
                fx, fy = p.force_x, p.force_y
                glyphs.append(VectorGlyph(p.id, p.x, p.y, fx * k, fy * k, (fx * fx + fy * fy) ** 0.5))  # type: ignore[operator]
        return glyphs

    # ------------------------------------------------------------- indicators
    def indicators(self) -> List[Indicator]:
        """Indicators reported by the simulation source.  Missing ones are ``unavailable``."""
        s = self._state
        if s is None:
            labels = ["Simulation time", "Timestep", "Step", "Particle count",
                      "Integration method", "Numerical precision", "Total energy", "Energy error"]
            return [Indicator(label, None) for label in labels]
        items = [
            Indicator("Simulation time", _fmt(s.simulation_time)),
            Indicator("Timestep", _fmt(s.timestep)),
            Indicator("Step", None if s.step is None else str(s.step)),
            Indicator("Particle count", str(s.particle_count)),  # a property of the snapshot itself
            Indicator("Integration method", s.integration_method),
            Indicator("Numerical precision", s.numerical_precision),
            Indicator("Total energy", _fmt(s.total_energy)),
            Indicator("Energy error", _fmt(s.energy_error)),
        ]
        for key in sorted(s.extras):
            value = s.extras[key]
            items.append(Indicator(key, value if isinstance(value, str) else _fmt(value)))
        return items

    def status_indicators(self) -> List[Indicator]:
        """Bookkeeping of the representation layer itself (not simulation data)."""
        items = [
            Indicator("Snapshots received", str(self.snapshots_received)),
            Indicator("Snapshots rejected", str(self.snapshots_rejected)),
            Indicator("Trajectory history", str(self.config.trajectory_history_length)),
        ]
        if self.history_resets:
            items.append(Indicator("History resets", str(self.history_resets)))
        if self.last_error:
            items.append(Indicator("Last rejection", self.last_error))
        return items
