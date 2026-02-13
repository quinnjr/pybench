from __future__ import annotations

import gc
import time

from pybench.results import BenchmarkResult

_DEFAULT_TARGET_NS = 1_000_000_000  # 1 second


class Runner:
    def __init__(
        self,
        warmup: int = 5,
        iterations: int | None = None,
        target_time_ns: int = _DEFAULT_TARGET_NS,
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.target_time_ns = target_time_ns

    def _calibrate(self, fn: callable) -> int:
        count = 1
        while True:
            start = time.perf_counter_ns()
            for _ in range(count):
                fn()
            elapsed = time.perf_counter_ns() - start
            if elapsed >= self.target_time_ns:
                return max(count, 1)
            count *= 2

    def run(self, name: str, fn: callable) -> BenchmarkResult:
        # Warmup
        for _ in range(self.warmup):
            fn()

        iterations = self.iterations if self.iterations is not None else self._calibrate(fn)

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            times: list[int] = []
            for _ in range(iterations):
                start = time.perf_counter_ns()
                fn()
                elapsed = time.perf_counter_ns() - start
                times.append(elapsed)
        finally:
            if gc_was_enabled:
                gc.enable()

        return BenchmarkResult.from_times(name, times)
