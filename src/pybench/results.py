from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    times_ns: list[int]
    iterations: int
    mean_ns: float
    median_ns: float
    stddev_ns: float
    min_ns: int
    max_ns: int
    ops_per_sec: float

    @classmethod
    def from_times(cls, name: str, times_ns: list[int]) -> BenchmarkResult:
        n = len(times_ns)
        mean = statistics.mean(times_ns)
        median = statistics.median(times_ns)
        stddev = statistics.stdev(times_ns) if n > 1 else 0.0
        ops = 1_000_000_000 / mean if mean > 0 else float("inf")

        return cls(
            name=name,
            times_ns=times_ns,
            iterations=n,
            mean_ns=mean,
            median_ns=median,
            stddev_ns=stddev,
            min_ns=min(times_ns),
            max_ns=max(times_ns),
            ops_per_sec=ops,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_ns": self.mean_ns,
            "median_ns": self.median_ns,
            "stddev_ns": self.stddev_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "ops_per_sec": self.ops_per_sec,
        }
