"""Tests for the GUI-free representation layer."""
import pytest

from psamathe_p12 import (IterableStateProvider, ParticleState, RepresentationConfig,
                          SimulationState, StateRepresentation, StateValidationError)


def state(t, positions, dt=0.1, **kw):
    particles = [ParticleState(id=i, x=x, y=y, vx=1.0, vy=0.0, mass=1.0) for i, (x, y) in enumerate(positions)]
    return SimulationState(simulation_time=t, timestep=dt, particles=particles, **kw)


def test_receiving_a_snapshot():
    rep = StateRepresentation()
    assert rep.state is None
    s = state(0.0, [(0, 0), (1, 1)])
    rep.update(s)
    assert rep.state is s
    assert rep.snapshots_received == 1


def test_update_rejects_non_state_objects():
    rep = StateRepresentation()
    with pytest.raises(StateValidationError):
        rep.update({"simulation_time": 0})
    assert rep.try_update("garbage") is False
    assert rep.snapshots_rejected == 1 and rep.last_error


def test_trajectory_history_update():
    rep = StateRepresentation()
    for k in range(4):
        rep.update(state(k * 0.1, [(k, 0)]))
    assert rep.trajectory(0) == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    assert rep.trajectory("unknown") == []


def test_trajectory_history_limit_bounds_memory():
    rep = StateRepresentation(RepresentationConfig(trajectory_history_length=5))
    for k in range(1000):
        rep.update(state(k * 0.1, [(k, 0), (0, k)]))
    assert len(rep.trajectory(0)) == 5
    assert rep.trajectory_point_count() == 10           # 2 particles x 5, not 2000
    assert rep.trajectory(0)[-1] == (999.0, 0.0)         # newest kept
    assert rep.trajectory(0)[0] == (995.0, 0.0)          # oldest dropped


def test_history_length_zero_disables_trails():
    rep = StateRepresentation(RepresentationConfig(trajectory_history_length=0))
    rep.update(state(0.0, [(0, 0)]))
    rep.update(state(0.1, [(1, 0)]))
    assert rep.trajectory_point_count() == 0


def test_config_validation():
    with pytest.raises(ValueError):
        RepresentationConfig(trajectory_history_length=-1)
    with pytest.raises(ValueError):
        RepresentationConfig(trajectory_history_length=10**9)   # hard ceiling
    with pytest.raises(ValueError):
        RepresentationConfig(velocity_scale=0)
    with pytest.raises(ValueError):
        RepresentationConfig(axis_mode="fixed")                 # needs fixed_extent


def test_removed_particles_free_their_history():
    rep = StateRepresentation()
    rep.update(state(0.0, [(0, 0), (1, 1)]))
    rep.update(state(0.1, [(0, 0)]))
    assert rep.trajectory(1) == []


def test_time_going_backwards_resets_history():
    rep = StateRepresentation()
    for k in range(3):
        rep.update(state(k * 0.1, [(k, 0)]))
    rep.update(state(0.0, [(9, 9)]))                  # restart
    assert rep.trajectory(0) == [(9.0, 9.0)]
    assert rep.history_resets == 1


def test_repeated_snapshot_does_not_duplicate_trail_points():
    rep = StateRepresentation()
    s = state(0.5, [(1, 1)])
    rep.update(s)
    rep.update(s)
    assert len(rep.trajectory(0)) == 1


def test_empty_particle_list_is_handled():
    rep = StateRepresentation()
    rep.update(state(0.0, []))
    assert rep.velocity_vectors() == [] and rep.force_vectors() == []
    assert rep.view_bounds() == (-1.0, 1.0, -1.0, 1.0)
    assert {i.label: i.display for i in rep.indicators()}["Particle count"] == "0"


def test_indicators_before_any_snapshot_are_unavailable():
    assert all(not i.available for i in StateRepresentation().indicators())


def test_optional_indicators_unavailable_not_fabricated():
    rep = StateRepresentation()
    rep.update(state(1.0, [(0, 0)]))
    ind = {i.label: i for i in rep.indicators()}
    assert ind["Simulation time"].display == "1"
    assert ind["Timestep"].display == "0.1"
    assert ind["Particle count"].display == "1"
    for label in ("Total energy", "Energy error", "Integration method", "Numerical precision", "Step"):
        assert not ind[label].available and ind[label].display == "unavailable"


def test_optional_indicators_shown_when_provided():
    rep = StateRepresentation()
    rep.update(state(1.0, [(0, 0)], step=10, integration_method="leapfrog", numerical_precision="float32",
                     total_energy=-0.5, energy_error=1e-6, extras={"wall_ms": 2.5, "note": "ok"}))
    ind = {i.label: i.display for i in rep.indicators()}
    assert ind["Step"] == "10"
    assert ind["Integration method"] == "leapfrog"
    assert ind["Numerical precision"] == "float32"
    assert ind["Total energy"] == "-0.5"
    assert ind["Energy error"] == "1e-06"
    assert ind["wall_ms"] == "2.5" and ind["note"] == "ok"


def test_velocity_vectors_explicit_scale():
    rep = StateRepresentation(RepresentationConfig(velocity_scale=0.5))
    rep.update(state(0.0, [(2, 3)]))
    g = rep.velocity_vectors()[0]
    assert (g.x, g.y, g.dx, g.dy, g.magnitude) == (2.0, 3.0, 0.5, 0.0, 1.0)


@pytest.mark.parametrize("speed", [1e-6, 1.0, 1e6])
def test_velocity_vectors_auto_scale_keeps_arrows_readable(speed):
    rep = StateRepresentation(RepresentationConfig(auto_vector_fraction=0.1, axis_margin=0.0))
    particles = [ParticleState(id=0, x=0, y=0, vx=speed, vy=0, mass=1),
                 ParticleState(id=1, x=10, y=10, vx=0, vy=speed, mass=1)]
    rep.update(SimulationState(0.0, 0.1, particles))
    longest = max(abs(g.dx) + abs(g.dy) for g in rep.velocity_vectors())
    assert longest == pytest.approx(1.0)               # 10% of the 10-unit span, whatever the speed


def test_force_vectors_only_for_particles_with_forces():
    rep = StateRepresentation()
    particles = [ParticleState(id=0, x=0, y=0, vx=0, vy=0, mass=1, force_x=1.0, force_y=0.0),
                 ParticleState(id=1, x=1, y=1, vx=0, vy=0, mass=1)]
    rep.update(SimulationState(0.0, 0.1, particles))
    assert [g.id for g in rep.force_vectors()] == [0]


def test_view_bounds_modes():
    expand = StateRepresentation(RepresentationConfig(axis_mode="expand", axis_margin=0.0))
    fit = StateRepresentation(RepresentationConfig(axis_mode="fit", axis_margin=0.0, trajectory_history_length=0))
    for r in (expand, fit):
        r.update(state(0.0, [(0, 0), (10, 0)]))
        r.update(state(0.1, [(0, 0), (2, 0)]))
    assert expand.view_bounds()[1] - expand.view_bounds()[0] == pytest.approx(10.0)   # remembers extent
    assert fit.view_bounds()[1] - fit.view_bounds()[0] == pytest.approx(2.0)          # follows data
    fixed = StateRepresentation(RepresentationConfig(axis_mode="fixed", fixed_extent=(-5, 5, -5, 5)))
    assert fixed.view_bounds() == (-5, 5, -5, 5)


def test_poll_counts_and_survives_malformed_snapshots():
    def stream():
        yield state(0.0, [(0, 0)])
        yield {"simulation_time": 1.0}                 # malformed mapping (no particles)
        yield state(0.2, [(1, 0)])

    provider = IterableStateProvider(stream())
    rep = StateRepresentation()
    results = [rep.poll(provider) for _ in range(4)]
    assert results == [True, False, True, False]
    assert rep.snapshots_received == 2 and rep.snapshots_rejected == 1
    assert "particles" in rep.last_error
    assert rep.state.simulation_time == 0.2            # last good state retained
    assert provider.finished


def test_representation_module_has_no_gui_or_mock_dependency():
    import subprocess, sys
    code = ("import sys, psamathe_p12.representation; "
            "assert 'matplotlib' not in sys.modules and 'psamathe_p12.mock_provider' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
