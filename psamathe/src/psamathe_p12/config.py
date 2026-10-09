"""Display/representation configuration for P1.2 (no simulation parameters)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# Hard ceiling so a typo cannot cause uncontrolled memory growth.
MAX_TRAJECTORY_HISTORY = 100_000

AXIS_MODES = ("expand", "fit", "fixed")


@dataclass(frozen=True)
class RepresentationConfig:
    """All tunable display behaviour.  None of this affects the simulation."""

    # Trajectories -----------------------------------------------------------
    trajectory_history_length: int = 300   # points kept per particle; 0 disables
    show_trajectories: bool = True

    # Velocity vectors -------------------------------------------------------
    show_velocity_vectors: bool = True
    velocity_scale: Optional[float] = None  # display length per unit velocity; None = auto
    auto_vector_fraction: float = 0.12      # auto mode: longest vector = this * axis span

    # Optional force vectors (drawn only for particles that supply forces) ----
    show_forces: bool = False
    force_scale: Optional[float] = None

    # Labels / markers -------------------------------------------------------
    show_particle_ids: bool = True
    marker_size_min: float = 25.0
    marker_size_max: float = 220.0

    # Axes -------------------------------------------------------------------
    axis_mode: str = "expand"               # expand | fit | fixed
    axis_margin: float = 0.10               # padding as a fraction of the span
    fixed_extent: Optional[Tuple[float, float, float, float]] = None  # xmin, xmax, ymin, ymax

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_history_length, int) or isinstance(self.trajectory_history_length, bool):
            raise ValueError("trajectory_history_length must be an int")
        if not 0 <= self.trajectory_history_length <= MAX_TRAJECTORY_HISTORY:
            raise ValueError(
                f"trajectory_history_length must be in [0, {MAX_TRAJECTORY_HISTORY}]"
            )
        for name in ("velocity_scale", "force_scale"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"{name} must be > 0 or None (auto)")
        if not 0 < self.auto_vector_fraction <= 1:
            raise ValueError("auto_vector_fraction must be in (0, 1]")
        if not 0 < self.marker_size_min <= self.marker_size_max:
            raise ValueError("require 0 < marker_size_min <= marker_size_max")
        if self.axis_mode not in AXIS_MODES:
            raise ValueError(f"axis_mode must be one of {AXIS_MODES}")
        if self.axis_margin < 0:
            raise ValueError("axis_margin must be >= 0")
        if self.axis_mode == "fixed":
            ext = self.fixed_extent
            if ext is None or len(ext) != 4 or not (ext[0] < ext[1] and ext[2] < ext[3]):
                raise ValueError("axis_mode='fixed' requires fixed_extent=(xmin, xmax, ymin, ymax)")
