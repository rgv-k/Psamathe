"""Provider / adapter compatibility tests, including the mock source."""
import pytest

from psamathe_p12 import (CallableStateProvider, IterableStateProvider, ParticleState, SimulationState,
                          SimulationStateProvider, StateRepresentation, StateValidationError, coerce_state)
from psamathe_p12.mock_provider import MockConfig, MockStateProvider


def test_mock_provider_satisfies_protocol():
    assert isinstance(MockStateProvider(), SimulationStateProvider)


def test_adapters_satisfy_protocol():
    assert isinstance(CallableStateProvider(lambda: None), SimulationStateProvider)
    assert isinstance(IterableStateProvider([]), SimulationStateProvider)


def test_mock_is_deterministic():
    a, b = MockStateProvider(MockConfig(seed=7)), MockStateProvider(MockConfig(seed=7))
    for _ in range(20):
        assert a.next_snapshot() == b.next_snapshot()
    assert MockStateProvider(MockConfig(seed=7)).next_snapshot() != MockStateProvider(MockConfig(seed=8)).next_snapshot()


def test_mock_snapshots_are_valid_and_honest_about_missing_data():
    p = MockStateProvider(MockConfig(n_particles=5, timestep=0.05))
    s0, s1 = p.next_snapshot(), p.next_snapshot()
    assert isinstance(s0, SimulationState) and s0.particle_count == 5
    assert s1.simulation_time - s0.simulation_time == pytest.approx(0.05)
    assert s0.total_energy is None and s0.energy_error is None     # never fabricated
    assert "Demo / Mock" in s0.extras["source"]


def test_mock_velocity_matches_position_derivative():
    """Demo velocities must be consistent with the demo paths (finite-difference check)."""
    p = MockStateProvider(MockConfig(n_particles=3, timestep=1e-6))
    a, b = p.next_snapshot(), p.next_snapshot()
    for pa, pb in zip(a.particles, b.particles):
        assert (pb.x - pa.x) / 1e-6 == pytest.approx(pa.vx, rel=1e-3, abs=1e-5)
        assert (pb.y - pa.y) / 1e-6 == pytest.approx(pa.vy, rel=1e-3, abs=1e-5)


def test_mock_finishes_and_closes():
    p = MockStateProvider(MockConfig(max_steps=3))
    got = []
    while not p.finished:
        got.append(p.next_snapshot())
    assert len(got) == 3 and p.next_snapshot() is None
    q = MockStateProvider()
    q.close()
    assert q.finished and q.next_snapshot() is None


def test_mock_config_validation():
    with pytest.raises(ValueError):
        MockConfig(timestep=0)
    with pytest.raises(ValueError):
        MockConfig(inner_radius=5, outer_radius=1)


def test_mock_single_and_zero_particles():
    assert MockStateProvider(MockConfig(n_particles=1)).next_snapshot().particle_count == 1
    assert MockStateProvider(MockConfig(n_particles=0)).next_snapshot().particle_count == 0


def test_representation_works_with_mock_through_the_interface_only():
    rep, p = StateRepresentation(), MockStateProvider(MockConfig(max_steps=10))
    while not p.finished:
        rep.poll(p)
    assert rep.snapshots_received == 10 and rep.state.step == 9


# ---- stand-in for the teammate's P1.1 object: a plain attribute holder, NO physics ----
class FakeP11:
    def __init__(self):
        self.time, self.dt, self.n_steps = 0.0, 0.01, 0
        self.bodies = [{"idx": 0, "pos": (0.0, 0.0), "vel": (1.0, 0.0), "m": 2.0},
                       {"idx": 1, "pos": (1.0, 1.0), "vel": (0.0, 1.0), "m": 1.0}]

    def advance(self):
        self.time += self.dt
        self.n_steps += 1


def convert_fake_p11(sim: FakeP11) -> SimulationState:
    return SimulationState(
        simulation_time=sim.time, timestep=sim.dt, step=sim.n_steps,
        particles=[ParticleState(id=b["idx"], x=b["pos"][0], y=b["pos"][1],
                                 vx=b["vel"][0], vy=b["vel"][1], mass=b["m"]) for b in sim.bodies])


def test_callable_adapter_wraps_a_foreign_simulator_object():
    sim = FakeP11()

    def fetch():
        sim.advance()
        return sim                     # foreign object, translated by `convert`

    provider = CallableStateProvider(fetch, convert=convert_fake_p11, is_finished=lambda: sim.n_steps >= 5)
    assert isinstance(provider, SimulationStateProvider)
    rep = StateRepresentation()
    while not provider.finished:
        assert rep.poll(provider)
    assert rep.snapshots_received == 5
    assert rep.state.step == 5 and rep.state.particle_count == 2


def test_callable_adapter_none_means_no_new_snapshot():
    provider = CallableStateProvider(lambda: None)
    assert provider.next_snapshot() is None
    assert StateRepresentation().poll(provider) is False


def test_callable_adapter_accepts_mappings_and_closes():
    closed = []
    provider = CallableStateProvider(lambda: {"simulation_time": 0.0, "particles": []},
                                     on_close=lambda: closed.append(True))
    assert provider.next_snapshot().particle_count == 0
    provider.close()
    assert closed == [True]


def test_coerce_state_errors():
    with pytest.raises(StateValidationError):
        coerce_state(42)
    with pytest.raises(StateValidationError):
        coerce_state(object(), convert=lambda o: "not a state")
