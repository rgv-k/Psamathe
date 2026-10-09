

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from .state import SimulationState, StateValidationError, state_from_mapping


def coerce_state(obj: Any, convert: Optional[Callable[[Any], SimulationState]] = None) -> SimulationState:
    """Convert ``obj`` into a ``SimulationState``.

    * ``SimulationState``      -> returned as is
    * ``Mapping``              -> ``state_from_mapping``
    * anything else            -> ``convert(obj)`` if a converter was supplied
    """
    if isinstance(obj, SimulationState):
        return obj
    if convert is not None:
        result = convert(obj)
        if not isinstance(result, SimulationState):
            raise StateValidationError(
                f"converter returned {type(result).__name__}, expected SimulationState"
            )
        return result
    if isinstance(obj, Mapping):
        return state_from_mapping(obj)
    raise StateValidationError(
        f"cannot interpret {type(obj).__name__} as a snapshot; pass a SimulationState, "
        "a mapping, or supply a converter"
    )


class CallableStateProvider:
    """Wrap a zero-argument function as a ``SimulationStateProvider``.

    Parameters
    ----------
    fetch:
        Called once per ``next_snapshot()``.  Returns a ``SimulationState``, a
        mapping, any object your ``convert`` function understands, or ``None``
        for "nothing new".
    convert:
        Optional function mapping whatever ``fetch`` returns to a
        ``SimulationState`` (this is where P1.1's own state object is
        translated into the P1.2 schema).
    is_finished:
        Optional zero-argument function reporting whether the source is done.
    on_close:
        Optional cleanup callback.
    """

    def __init__(
        self,
        fetch: Callable[[], Any],
        convert: Optional[Callable[[Any], SimulationState]] = None,
        is_finished: Optional[Callable[[], bool]] = None,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self._fetch = fetch
        self._convert = convert
        self._is_finished = is_finished
        self._on_close = on_close

    @property
    def finished(self) -> bool:
        return bool(self._is_finished()) if self._is_finished else False

    def next_snapshot(self) -> Optional[SimulationState]:
        raw = self._fetch()
        if raw is None:
            return None
        return coerce_state(raw, self._convert)

    def close(self) -> None:
        if self._on_close:
            self._on_close()


class IterableStateProvider:
    """Wrap any iterable of snapshots (states or mappings), e.g. a recorded run."""

    def __init__(
        self,
        snapshots: Iterable[Any],
        convert: Optional[Callable[[Any], SimulationState]] = None,
    ) -> None:
        self._iterator: Iterator[Any] = iter(snapshots)
        self._convert = convert
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def next_snapshot(self) -> Optional[SimulationState]:
        try:
            raw = next(self._iterator)
        except StopIteration:
            self._finished = True
            return None
        return coerce_state(raw, self._convert)

    def close(self) -> None:
        self._finished = True
