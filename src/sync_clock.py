"""A shared timestamp source for visual and audio capture."""

from __future__ import annotations

import threading
import time


class SyncClock:
    """Provide wall-clock timestamps from one shared clock instance.

    ``time.time()`` is used for exported timestamps so CSV files are easy to
    correlate with other experiments. ``time.perf_counter()`` is retained for
    stable elapsed-time measurements during a capture session.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wall_start = time.time()
        self._perf_start = time.perf_counter()

    def now(self) -> float:
        """Return the current Unix timestamp in seconds."""
        with self._lock:
            return self._wall_start + (time.perf_counter() - self._perf_start)

    def elapsed(self) -> float:
        """Return seconds elapsed since this clock was created."""
        return time.perf_counter() - self._perf_start


_DEFAULT_CLOCK = SyncClock()


def get_timestamp() -> float:
    """Return a timestamp from the process-wide default clock."""
    return _DEFAULT_CLOCK.now()

