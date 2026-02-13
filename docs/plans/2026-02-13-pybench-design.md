# pybench Design

A pip-installable Python benchmarking library with CLI. Zero runtime dependencies for core functionality, `rich` optional for pretty terminal output. Targets Python 3.10+.

## Core API

Two ways to define benchmarks:

**Decorator:**
```python
import pybench

@pybench.benchmark
def my_sort():
    sorted([3, 1, 2] * 1000)

@pybench.benchmark(warmup=5, iterations=100)
def my_hash():
    hash("hello" * 1000)
```

**Context manager:**
```python
from pybench import Bench

bench = Bench()

with bench.measure("list comprehension"):
    [x ** 2 for x in range(1000)]

with bench.measure("map"):
    list(map(lambda x: x ** 2, range(1000)))

bench.report()
```

The `Bench` class is the central orchestrator. It collects benchmark functions (via decorator) or inline measurements (via context manager), runs them with configurable warmup/iterations, and produces `BenchmarkResult` objects.

## Data Model

```
BenchmarkResult:
  - name: str
  - times_ns: list[int]       # raw timings in nanoseconds
  - iterations: int
  - mean_ns: float
  - median_ns: float
  - stddev_ns: float
  - min_ns: int
  - max_ns: int
  - ops_per_sec: float
```

## Execution Engine

- Uses `time.perf_counter_ns()` for nanosecond precision
- Configurable warmup rounds (default: 5) — runs discarded before measurement
- Configurable iterations (default: auto-calibrate to ~1 second of total runtime)
- Auto-calibration: runs the function repeatedly, doubling count until total time exceeds a target threshold, then uses that count
- GC disabled during measurement, re-enabled between benchmarks

## Output

**Terminal (default):** A formatted table using `rich` if installed, plain-text aligned columns otherwise:
```
pybench results
───────────────────────────────────────────────────────────────
Name               Mean        Median      StdDev     Ops/sec
───────────────────────────────────────────────────────────────
list comprehension  45.2 µs     44.8 µs     1.2 µs    22,124
map                 52.1 µs     51.9 µs     0.8 µs    19,194
───────────────────────────────────────────────────────────────
```

**JSON:** `--json` flag or `bench.to_json()` produces structured output for CI:
```json
{
  "metadata": {"python_version": "3.12.1", "platform": "Linux", "timestamp": "..."},
  "results": [{"name": "...", "mean_ns": ..., ...}]
}
```

## Comparison

`pybench compare baseline.json current.json` — reads two JSON result files and prints a diff table showing % change per benchmark, with color-coded regression/improvement indicators.

## CLI

```
pybench run [PATH]           # discover and run bench_*.py files (or specific file)
pybench run --json           # output JSON instead of table
pybench run --save FILE      # run and save JSON results
pybench compare A.json B.json  # compare two result sets
```

Discovery: looks for files matching `bench_*.py` or `*_bench.py` in the given path, imports them, and runs any `@pybench.benchmark`-decorated functions found.

## Project Structure

```
pybench/
├── pyproject.toml
├── src/
│   └── pybench/
│       ├── __init__.py      # public API re-exports
│       ├── bench.py         # Bench class, decorator, context manager
│       ├── runner.py        # execution engine (timing, warmup, calibration)
│       ├── results.py       # BenchmarkResult dataclass
│       ├── reporter.py      # terminal + JSON output
│       ├── compare.py       # comparison logic
│       └── cli.py           # argparse CLI
└── tests/
    ├── test_bench.py
    ├── test_runner.py
    ├── test_reporter.py
    └── test_compare.py
```

Uses `pyproject.toml` with setuptools, src layout, `[project.scripts]` for the `pybench` CLI entry point. `rich` listed as optional dependency under `[project.optional-dependencies]`.

## Approach

Lightweight stdlib-only core. Zero dependencies for core library. Uses `time.perf_counter_ns` for timing, `statistics` for stats. `rich` is an optional dependency for pretty terminal output (falls back to plain text tables). `argparse` for CLI.
