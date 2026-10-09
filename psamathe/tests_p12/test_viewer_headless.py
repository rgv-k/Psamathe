"""Smoke tests of the matplotlib viewer on the non-interactive Agg backend (no window opens)."""
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from psamathe_p12 import ParticleState, RepresentationConfig, SimulationState, StateRepresentation  # noqa: E402
from psamathe_p12.mock_provider import MockConfig, MockStateProvider  # noqa: E402
from psamathe_p12.viewer import MatplotlibViewer, render_to_file  # noqa: E402


def test_viewer_draws_every_state_without_error():
    import matplotlib.pyplot as plt
    rep = StateRepresentation(RepresentationConfig(show_forces=True))
    viewer = MatplotlibViewer(rep, "test")
    viewer.draw()                                            # before any snapshot
    rep.update(SimulationState(0.0, 0.1, []))
    viewer.draw()                                            # empty particle list
    for k in range(3):
        ps = [ParticleState(id=i, x=i + k, y=0, vx=1, vy=0, mass=1 + i, force_x=0.1, force_y=0.1) for i in range(3)]
        rep.update(SimulationState(k * 0.1, 0.1, ps))
        viewer.draw()
    text = viewer._panel_text.get_text()
    assert "Total energy: unavailable" in text and "Particle count: 3" in text
    plt.close(viewer.fig)


def test_render_to_png(tmp_path):
    out = tmp_path / "frame.png"
    rep = render_to_file(MockStateProvider(MockConfig(max_steps=20)), None, out, frames=50)
    assert out.exists() and out.stat().st_size > 1000
    assert rep.snapshots_received == 20
