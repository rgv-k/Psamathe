from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .state import SimulationState


@runtime_checkable
class SimulationStateProvider(Protocol):
    """Anything that can hand P1.2 a stream of ``SimulationState`` snapshots.

    Contract
    --------
    ``next_snapshot()``
        Return the next available snapshot, or ``None`` if there is nothing
        new right now.  It must return ``SimulationState`` objects (use
        ``state_from_mapping`` inside your adapter if you have dicts).  It may
        raise ``StateValidationError`` for a malformed snapshot; P1.2 then
        rejects that snapshot, counts it, and keeps showing the last good one.

    ``finished``
        ``True`` once the source will never produce another snapshot.

    ``close()``
        Release resources.  Called once when the viewer shuts down.

    The provider decides *when* new state exists (for example, by stepping the
    simulator inside ``next_snapshot``, or by returning the latest state the
    simulator has published).  P1.2 never sends commands back: there is
    deliberately no method here to change the timestep, precision or physics.
    """

    @property
    def finished(self) -> bool: ...

    def next_snapshot(self) -> Optional[SimulationState]: ...

    def close(self) -> None: ...
