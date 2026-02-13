from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from pybench.results import BenchmarkResult
from pybench.runner import Runner

# Global registry for module-level @benchmark decorator
_global_registry: list[tuple[str, callable, dict]] = []


class Bench:
    def __init__(
        self,
        warmup: int = 5,
        iterations: int | None = None,
        target_time_ns: int = 1_000_000_000,
    ):
        self._warmup = warmup
        self._iterations = iterations
        self._target_time_ns = target_time_ns
        self._registered: list[tuple[str, callable, dict[str, Any]]] = []
        self._results: list[BenchmarkResult] = []

    def benchmark(self, fn=None, /, *, warmup=None, iterations=None):
        """Decorator to register a benchmark function.

        Can be used as @bench.benchmark or @bench.benchmark(warmup=2, iterations=10).
        """
        opts = {}
        if warmup is not None:
            opts["warmup"] = warmup
        if iterations is not None:
            opts["iterations"] = iterations

        if fn is not None:
            # Used as @bench.benchmark (no parentheses)
            self._registered.append((fn.__name__, fn, opts))
            return fn

        # Used as @bench.benchmark(warmup=2, iterations=10)
        def decorator(f):
            self._registered.append((f.__name__, f, opts))
            return f
        return decorator

    @contextmanager
    def measure(self, name: str):
        """Context manager for inline benchmarks."""
        import time
        start = time.perf_counter_ns()
        yield
        elapsed = time.perf_counter_ns() - start
        result = BenchmarkResult.from_times(name, [elapsed])
        self._results.append(result)

    def run(self) -> list[BenchmarkResult]:
        """Run all registered benchmarks and return results."""
        results = list(self._results)  # include context manager results

        for name, fn, opts in self._registered:
            runner = Runner(
                warmup=opts.get("warmup", self._warmup),
                iterations=opts.get("iterations", self._iterations),
                target_time_ns=self._target_time_ns,
            )
            result = runner.run(name, fn)
            results.append(result)

        self._results = results
        return results

    def report(self, json_output: bool = False) -> None:
        """Print benchmark results to stdout."""
        from pybench.reporter import format_json, format_table

        if not self._results:
            self.run()

        if json_output:
            print(format_json(self._results))
        else:
            print(format_table(self._results))

    def to_json(self) -> str:
        """Return benchmark results as a JSON string."""
        from pybench.reporter import format_json

        if not self._results:
            self.run()

        return format_json(self._results)


def benchmark(fn=None, /, *, warmup=None, iterations=None):
    """Module-level decorator that registers benchmarks in the global registry."""
    opts = {}
    if warmup is not None:
        opts["warmup"] = warmup
    if iterations is not None:
        opts["iterations"] = iterations

    if fn is not None:
        _global_registry.append((fn.__name__, fn, opts))
        return fn

    def decorator(f):
        _global_registry.append((f.__name__, f, opts))
        return f
    return decorator
