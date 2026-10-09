

from __future__ import annotations

import itertools
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from .config import RepresentationConfig
from .provider import SimulationStateProvider
from .representation import StateRepresentation

_PALETTE = plt.get_cmap("tab20")


class MatplotlibViewer:
    """Draws particles, trajectories, velocity vectors and indicators."""

    def __init__(self, representation: StateRepresentation, source_label: str = "unknown source") -> None:
        self.rep = representation
        self.source_label = source_label
        cfg = representation.config
        self.show_trajectories = cfg.show_trajectories
        self.show_velocity = cfg.show_velocity_vectors
        self.show_ids = cfg.show_particle_ids
        self.paused = False
        self.animation: Optional[FuncAnimation] = None
        self._colors: Dict[object, int] = {}

        self.fig = plt.figure(figsize=(11.5, 6.6))
        self.ax = self.fig.add_axes([0.06, 0.08, 0.60, 0.84])
        self.panel = self.fig.add_axes([0.69, 0.08, 0.30, 0.84])
        self.panel.axis("off")

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x (simulation units)")
        self.ax.set_ylabel("y (simulation units)")
        self.ax.grid(True, alpha=0.25)

        self._lines = LineCollection([], linewidths=1.1, alpha=0.55, zorder=1)
        self.ax.add_collection(self._lines)
        self._scatter = self.ax.scatter([], [], s=[], zorder=4, edgecolors="black", linewidths=0.5)
        self._velocity_quiver = None
        self._force_quiver = None
        self._labels: List = []
        self._panel_text = self.panel.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.5)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ---------------------------------------------------------------- helpers
    def _color(self, pid: object):
        idx = self._colors.setdefault(pid, len(self._colors))
        return _PALETTE(idx % _PALETTE.N)

    def _marker_sizes(self, masses: np.ndarray) -> np.ndarray:
        cfg = self.rep.config
        if masses.size == 0:
            return masses
        frac = np.sqrt(masses / masses.max())
        return cfg.marker_size_min + (cfg.marker_size_max - cfg.marker_size_min) * frac

    def _replace_quiver(self, old, glyphs, color: str, zorder: int):
        if old is not None:
            old.remove()
        if not glyphs:
            return None
        arr = np.array([(g.x, g.y, g.dx, g.dy) for g in glyphs], dtype=float)
        # Glyph components are already in data units -> scale_units='xy', scale=1.
        return self.ax.quiver(
            arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
            angles="xy", scale_units="xy", scale=1.0, color=color, width=0.003, zorder=zorder,
        )

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "t":
            self.show_trajectories = not self.show_trajectories
        elif event.key == "v":
            self.show_velocity = not self.show_velocity
        elif event.key == "i":
            self.show_ids = not self.show_ids

    # ------------------------------------------------------------------- draw
    def draw(self, stream_ended: bool = False) -> None:
        """Redraw everything from the representation's current state."""
        rep, cfg = self.rep, self.rep.config
        state = rep.state
        particles = state.particles if state is not None else ()

        xy = np.array([(p.x, p.y) for p in particles], dtype=float).reshape(-1, 2)
        colors = [self._color(p.id) for p in particles]
        masses = np.array([p.mass for p in particles], dtype=float)
        self._scatter.set_offsets(xy)
        self._scatter.set_sizes(self._marker_sizes(masses))
        if colors:
            self._scatter.set_facecolor(colors)

        segments, seg_colors = [], []
        if self.show_trajectories:
            for p, col in zip(particles, colors):
                pts = rep.trajectory(p.id)
                if len(pts) >= 2:
                    segments.append(np.asarray(pts, dtype=float))
                    seg_colors.append(col)
        self._lines.set_segments(segments)
        if seg_colors:
            self._lines.set_color(seg_colors)

        self._velocity_quiver = self._replace_quiver(
            self._velocity_quiver, rep.velocity_vectors() if self.show_velocity else [], "#222222", 3)
        self._force_quiver = self._replace_quiver(
            self._force_quiver, rep.force_vectors() if cfg.show_forces else [], "crimson", 2)

        for label in self._labels:
            label.remove()
        self._labels = []
        if self.show_ids:
            for p in particles:
                self._labels.append(self.ax.annotate(
                    str(p.id), (p.x, p.y), xytext=(5, 5), textcoords="offset points", fontsize=8, zorder=5))

        xmin, xmax, ymin, ymax = rep.view_bounds()
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        t = "n/a" if state is None else f"{state.simulation_time:.6g}"
        self.ax.set_title(f"Psamathe P1.2 - state representation   (t = {t})")
        self._panel_text.set_text(self._panel_string(stream_ended))

    def _panel_string(self, stream_ended: bool) -> str:
        rep = self.rep
        lines = ["NUMERICAL INDICATORS", "(from the state snapshot)", ""]
        lines += [f"{i.label}: {i.display}" for i in rep.indicators()]
        lines += ["", "REPRESENTATION STATUS"]
        lines += [f"{i.label}: {i.display}" for i in rep.status_indicators()]
        lines += ["", "LEGEND"]
        if self.show_velocity:
            lines.append(f"black arrow = velocity (x{rep.velocity_scale_in_use():.3g})")
        if rep.config.show_forces:
            lines.append("red arrow = force (if provided)")
        lines.append("marker size ~ sqrt(mass)")
        lines += ["", f"Source: {self.source_label}",
                  "Keys: SPACE pause | T trails | V vel | I ids"]
        if self.paused:
            lines.append("[PAUSED]")
        if stream_ended:
            lines.append("[STREAM ENDED]")
        # Wrap very long values (e.g. error text) so they stay inside the panel
        return "\n".join(_wrap(line, 44) for line in lines)


def _wrap(line: str, width: int) -> str:
    return textwrap.fill(line, width=width, subsequent_indent="  ", break_long_words=True) if line else line


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def run_viewer(
    provider: SimulationStateProvider,
    config: Optional[RepresentationConfig] = None,
    *,
    interval_ms: int = 40,
    frames: Optional[int] = None,
    source_label: Optional[str] = None,
    show: bool = True,
) -> StateRepresentation:
    """Open the live viewer and animate until the provider finishes (or window closes)."""
    rep = StateRepresentation(config)
    label = source_label or getattr(provider, "label", type(provider).__name__)
    viewer = MatplotlibViewer(rep, source_label=label)
    holder: List[FuncAnimation] = []

    def step(_frame: int) -> None:
        if not viewer.paused:
            rep.poll(provider)
        ended = provider.finished
        viewer.draw(stream_ended=ended)
        if ended and holder:
            holder[0].event_source.stop()

    viewer.animation = FuncAnimation(
        viewer.fig, step, frames=frames if frames is not None else itertools.count,
        interval=interval_ms, repeat=False, cache_frame_data=False,
    )
    holder.append(viewer.animation)
    try:
        if show:
            plt.show()
    finally:
        provider.close()
    return rep


def render_to_file(
    provider: SimulationStateProvider,
    config: Optional[RepresentationConfig],
    path: Union[str, Path],
    *,
    frames: int = 300,
    fps: int = 20,
    source_label: Optional[str] = None,
) -> StateRepresentation:
    """Headless rendering: ``.gif`` writes an animation, anything else a PNG of the final frame."""
    path = Path(path)
    rep = StateRepresentation(config)
    label = source_label or getattr(provider, "label", type(provider).__name__)
    viewer = MatplotlibViewer(rep, source_label=label)
    try:
        if path.suffix.lower() == ".gif":
            def step(_frame: int) -> None:
                rep.poll(provider)
                viewer.draw(stream_ended=provider.finished)

            anim = FuncAnimation(viewer.fig, step, frames=frames, interval=1000 // fps,
                                 repeat=False, cache_frame_data=False)
            anim.save(str(path), writer=PillowWriter(fps=fps))
        else:
            for _ in range(frames):
                rep.poll(provider)
                if provider.finished:
                    break
            viewer.draw(stream_ended=provider.finished)
            viewer.fig.savefig(str(path), dpi=150)
    finally:
        provider.close()
        plt.close(viewer.fig)
    return rep
