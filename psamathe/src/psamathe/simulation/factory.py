from psamathe.core.state import SimulationState
from psamathe.gui.state import ExperimentConfiguration
from psamathe.integrators import (
    EulerIntegrator,
    VelocityVerletIntegrator,
    RK4Integrator,
)
from psamathe.physics.gravity import compute_accelerations
from psamathe.simulation.engine import SimulationEngine


def create_simulation_engine(
    configuration: ExperimentConfiguration,
) -> SimulationEngine:
    """
    Build a SimulationEngine from a GUI experiment configuration.

    The GUI configuration is translated into the core simulation
    objects here so that the GUI itself does not contain physics logic.
    """

    masses = []
    positions = []
    velocities = []

    for body in configuration.bodies:
        masses.append(body["mass"])

        positions.append([
            body["x"],
            body["y"],
        ])

        velocities.append([
            body["vx"],
            body["vy"],
        ])

    state = SimulationState(
        mass=masses,
        position=positions,
        velocity=velocities,
        time=0.0,
    )

    integrators = {
        "Euler": EulerIntegrator(),
        "Velocity Verlet": VelocityVerletIntegrator(),
        "RK4": RK4Integrator(),
    }

    try:
        integrator = integrators[configuration.integrator]
    except KeyError:
        raise ValueError(
            f"Unsupported integrator: {configuration.integrator}"
        )

    return SimulationEngine(
        state=state,
        integrator=integrator,
        acceleration_function=compute_accelerations,
        dt=configuration.dt,
    )