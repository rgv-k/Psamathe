"""Template: how the real P1.1 simulator plugs into P1.2.

Run from the project root (uses a tiny stand-in object, NOT real physics):

    python examples/connect_p11_example.py

To connect the real simulator, replace ``StandInSimulator`` with your teammate's
class and adjust ``to_snapshot`` so it maps P1.1's fields onto the P1.2 schema.
Nothing inside the ``psamathe_p12`` package needs to change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psamathe_p12 import CallableStateProvider, ParticleState, RepresentationConfig, SimulationState
from psamathe_p12.viewer import run_viewer


class StandInSimulator:
    """Placeholder for P1.1.  Moves bodies in straight lines; it is NOT gravity."""

    def __init__(self):
        self.time, self.dt, self.n_steps = 0.0, 0.05, 0
        self.bodies = [dict(id=0, x=-1.0, y=0.0, vx=0.4, vy=0.1, m=1.0),
                       dict(id=1, x=1.0, y=0.5, vx=-0.3, vy=0.0, m=2.0)]

    def advance(self):
        for b in self.bodies:
            b["x"] += b["vx"] * self.dt
            b["y"] += b["vy"] * self.dt
        self.time += self.dt
        self.n_steps += 1


def to_snapshot(sim: StandInSimulator) -> SimulationState:
    """THE ADAPTER: translate P1.1's own state into the P1.2 schema (read-only)."""
    return SimulationState(
        simulation_time=sim.time,
        timestep=sim.dt,
        step=sim.n_steps,
        particles=[ParticleState(id=b["id"], x=b["x"], y=b["y"], vx=b["vx"], vy=b["vy"], mass=b["m"])
                   for b in sim.bodies],
        # integration_method / numerical_precision / total_energy: pass them only if P1.1 has them.
    )


if __name__ == "__main__":
    sim = StandInSimulator()

    def fetch():
        sim.advance()          # who advances the simulation is decided here, in the integration code
        return sim

    provider = CallableStateProvider(fetch, convert=to_snapshot, is_finished=lambda: sim.n_steps >= 400)
    run_viewer(provider, RepresentationConfig(trajectory_history_length=150), source_label="Stand-in (example only)")
