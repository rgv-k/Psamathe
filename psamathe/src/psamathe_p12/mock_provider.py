"""P1.2 Demo / Mock State Provider.

*** This is NOT the Psamathe physics simulator. ***

It exists only so the P1.2 visualization can be run and tested before P1.1 is
available.  It does not compute gravitational forces and does not use any
numerical integrator.  Instead it evaluates fixed, closed-form elliptical
paths (position and the exact analytic time-derivative for velocity) around a
stationary central body, purely to generate believable, deterministic data.

Replace it with a real P1.1 adapter (see README, "Connecting P1.1").  Nothing
in the representation or viewer modules imports this file.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from .state import ParticleState, SimulationState

MOCK_LABEL = "P1.2 Demo / Mock State Provider"


@dataclass(frozen=True)
class MockConfig:
    n_particles: int = 8          # includes the stationary central body
    seed: int = 42                # same seed -> identical stream
    timestep: float = 0.02        # simulation time advanced per snapshot
    max_steps: Optional[int] = None  # None = endless stream
    inner_radius: float = 1.0
    outer_radius: float = 4.0

    def __post_init__(self) -> None:
        if self.n_particles < 0:
            raise ValueError("n_particles must be >= 0")
        if self.timestep <= 0:
            raise ValueError("timestep must be > 0")
        if self.max_steps is not None and self.max_steps < 0:
            raise ValueError("max_steps must be >= 0 or None")
        if not 0 < self.inner_radius < self.outer_radius:
            raise ValueError("require 0 < inner_radius < outer_radius")


@dataclass(frozen=True)
class _Track:
    """Closed-form elliptical path parameters for one demo particle."""

    pid: int
    mass: float
    a: float        # semi-major axis
    b: float        # semi-minor axis
    tilt: float     # rotation of the ellipse
    omega: float    # angular rate
    phase: float


class MockStateProvider:
    """Deterministic demo stream implementing ``SimulationStateProvider``."""

    label = MOCK_LABEL

    def __init__(self, config: Optional[MockConfig] = None) -> None:
        self.config = config or MockConfig()
        rng = random.Random(self.config.seed)
        cfg = self.config
        self._tracks: List[_Track] = []
        for i in range(1, cfg.n_particles):
            a = cfg.inner_radius + (cfg.outer_radius - cfg.inner_radius) * (i - 1) / max(cfg.n_particles - 2, 1)
            self._tracks.append(
                _Track(
                    pid=i,
                    mass=round(rng.uniform(0.2, 1.0), 3),
                    a=a,
                    b=a * rng.uniform(0.6, 1.0),
                    tilt=rng.uniform(0.0, math.tau),
                    # Looks Kepler-like (faster when closer) but is just a formula.
                    omega=(1.0 if rng.random() < 0.8 else -1.0) * 2.0 / a ** 1.5,
                    phase=rng.uniform(0.0, math.tau),
                )
            )
        self._step = 0
        self._closed = False

    # -- SimulationStateProvider -------------------------------------------
    @property
    def finished(self) -> bool:
        cfg = self.config
        return self._closed or (cfg.max_steps is not None and self._step >= cfg.max_steps)

    def next_snapshot(self) -> Optional[SimulationState]:
        if self.finished:
            return None
        cfg = self.config
        t = self._step * cfg.timestep
        particles: List[ParticleState] = []
        if cfg.n_particles > 0:
            particles.append(ParticleState(id=0, x=0.0, y=0.0, vx=0.0, vy=0.0, mass=50.0))
        for tr in self._tracks:
            particles.append(self._evaluate(tr, t))
        state = SimulationState(
            simulation_time=t,
            timestep=cfg.timestep,
            particles=particles,
            step=self._step,
            integration_method="none (closed-form demo paths)",
            numerical_precision="float64",
            # total_energy / energy_error deliberately omitted: the mock has no
            # physics, so it must not invent them.  The viewer shows "unavailable".
            extras={"source": MOCK_LABEL},
        )
        self._step += 1
        return state

    def close(self) -> None:
        self._closed = True

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _evaluate(tr: _Track, t: float) -> ParticleState:
        theta = tr.omega * t + tr.phase
        ex, ey = tr.a * math.cos(theta), tr.b * math.sin(theta)
        evx, evy = -tr.a * tr.omega * math.sin(theta), tr.b * tr.omega * math.cos(theta)
        c, s = math.cos(tr.tilt), math.sin(tr.tilt)
        return ParticleState(
            id=tr.pid,
            x=c * ex - s * ey, y=s * ex + c * ey,
            vx=c * evx - s * evy, vy=s * evx + c * evy,
            mass=tr.mass,
        )
