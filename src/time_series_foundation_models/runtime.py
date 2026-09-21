"""Runtime and memory measurement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import tracemalloc
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RuntimeMeasurement:
    seconds: float
    peak_memory_mb: float


def measure_call(function: Callable[[], T]) -> tuple[T, RuntimeMeasurement]:
    """Execute a callable and measure wall-clock time and Python peak memory."""
    tracemalloc.start()
    started = perf_counter()
    try:
        result = function()
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    measurement = RuntimeMeasurement(
        seconds=float(elapsed),
        peak_memory_mb=float(peak / (1024**2)),
    )
    return result, measurement
