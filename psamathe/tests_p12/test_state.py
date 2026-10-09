"""Tests for the state model (no GUI, no matplotlib)."""
import pytest

from psamathe_p12 import ParticleState, SimulationState, StateValidationError, state_from_mapping


def make_particle(pid=1, **kw):
    base = dict(id=pid, x=1.0, y=2.0, vx=0.1, vy=-0.2, mass=3.0)
    base.update(kw)
    return ParticleState(**base)


def test_valid_simulation_state_creation():
    s = SimulationState(simulation_time=1.5, timestep=0.01, particles=[make_particle(1), make_particle(2)])
    assert s.particle_count == 2
    assert s.simulation_time == 1.5 and s.timestep == 0.01
    assert isinstance(s.particles, tuple)          # normalised to an immutable tuple
    assert s.total_energy is None and s.integration_method is None


def test_ints_are_coerced_to_float():
    p = ParticleState(id=1, x=1, y=2, vx=0, vy=0, mass=1)
    assert isinstance(p.x, float) and p.x == 1.0


def test_snapshot_is_immutable():
    s = SimulationState(0.0, 0.1, [make_particle()])
    with pytest.raises(Exception):
        s.simulation_time = 5.0           # frozen dataclass: P1.2 cannot alter state
    with pytest.raises(Exception):
        s.particles[0].x = 9.0


@pytest.mark.parametrize("field,value", [
    ("x", float("nan")), ("y", float("inf")), ("vx", "fast"), ("vy", None),
    ("mass", 0.0), ("mass", -1.0), ("x", True),
])
def test_particle_validation_rejects_bad_values(field, value):
    with pytest.raises(StateValidationError):
        make_particle(**{field: value})


@pytest.mark.parametrize("pid", [True, 1.5, "", "  ", None, [1]])
def test_particle_validation_rejects_bad_ids(pid):
    with pytest.raises(StateValidationError):
        make_particle(pid)


def test_force_components_must_come_together():
    with pytest.raises(StateValidationError):
        make_particle(force_x=1.0)
    assert make_particle(force_x=1.0, force_y=-1.0).has_force
    assert not make_particle().has_force


def test_speed_property():
    assert make_particle(vx=3.0, vy=4.0).speed == pytest.approx(5.0)


def test_duplicate_particle_ids_rejected():
    with pytest.raises(StateValidationError, match="duplicate"):
        SimulationState(0.0, 0.1, [make_particle(1), make_particle(1)])


@pytest.mark.parametrize("kwargs", [
    dict(simulation_time=float("nan"), timestep=0.1),
    dict(simulation_time=0.0, timestep=0.0),
    dict(simulation_time=0.0, timestep=-0.1),
    dict(simulation_time="0", timestep=0.1),
])
def test_simulation_state_rejects_bad_time_or_timestep(kwargs):
    with pytest.raises(StateValidationError):
        SimulationState(particles=[], **kwargs)


def test_particles_must_be_particle_states():
    with pytest.raises(StateValidationError):
        SimulationState(0.0, 0.1, [{"id": 1}])
    with pytest.raises(StateValidationError):
        SimulationState(0.0, 0.1, "not particles")


def test_optional_indicators_validated():
    with pytest.raises(StateValidationError):
        SimulationState(0.0, 0.1, [], total_energy=float("inf"))
    with pytest.raises(StateValidationError):
        SimulationState(0.0, 0.1, [], step=-1)
    with pytest.raises(StateValidationError):
        SimulationState(0.0, 0.1, [], integration_method=5)
    with pytest.raises(StateValidationError):
        SimulationState(0.0, 0.1, [], extras={"k": [1]})


def test_timestep_may_be_unavailable():
    assert SimulationState(0.0, None, []).timestep is None


def test_empty_particle_list_is_valid():
    assert SimulationState(0.0, 0.1, []).particle_count == 0


def test_state_from_mapping_roundtrip():
    original = SimulationState(
        2.0, 0.05, [make_particle(1, force_x=0.5, force_y=0.25), make_particle("b")],
        step=40, integration_method="leapfrog", numerical_precision="float64",
        total_energy=-1.25, energy_error=1e-9, extras={"cost_ms": 3.5})
    assert state_from_mapping(original.to_dict()) == original


def test_state_from_mapping_minimal_and_errors():
    s = state_from_mapping({"simulation_time": 0.0, "particles": []})
    assert s.timestep is None
    with pytest.raises(StateValidationError, match="simulation_time"):
        state_from_mapping({"particles": []})
    with pytest.raises(StateValidationError, match="index 0"):
        state_from_mapping({"simulation_time": 0.0, "particles": [{"id": 1, "x": 0}]})
    with pytest.raises(StateValidationError):
        state_from_mapping({"simulation_time": 0.0, "particles": "abc"})
    with pytest.raises(StateValidationError):
        state_from_mapping([1, 2, 3])
