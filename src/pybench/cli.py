from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from pybench.bench import Bench, _global_registry
from pybench.compare import format_comparison
from pybench.reporter import format_json, format_table
from pybench.runner import Runner


def _discover_benchmarks(path: Path) -> list[tuple[str, callable, dict]]:
    """Import bench_*.py / *_bench.py files and collect globally registered benchmarks."""
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            list(path.glob("bench_*.py")) + list(path.glob("*_bench.py"))
        )

    _global_registry.clear()

    for f in files:
        spec = importlib.util.spec_from_file_location(f.stem, f)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f.stem] = mod
            spec.loader.exec_module(mod)

    return list(_global_registry)


def _cmd_run(args: argparse.Namespace) -> None:
    path = Path(args.path)
    benchmarks = _discover_benchmarks(path)

    if not benchmarks:
        print("No benchmarks found.", file=sys.stderr)
        sys.exit(1)

    bench = Bench(
        warmup=args.warmup,
        iterations=args.iterations,
    )

    for name, fn, opts in benchmarks:
        runner = Runner(
            warmup=opts.get("warmup", args.warmup),
            iterations=opts.get("iterations", args.iterations),
        )
        result = runner.run(name, fn)
        bench._results.append(result)

    results = bench._results

    if args.json:
        output = format_json(results)
        print(output)
    else:
        print(format_table(results))

    if args.save:
        Path(args.save).write_text(format_json(results))


def _cmd_compare(args: argparse.Namespace) -> None:
    baseline = Path(args.baseline).read_text()
    current = Path(args.current).read_text()
    print(format_comparison(baseline, current))


def main() -> None:
    parser = argparse.ArgumentParser(prog="pybench", description="Python benchmarking tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("path", nargs="?", default=".", help="Path to benchmark file or directory")
    run_parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    run_parser.add_argument("--save", metavar="FILE", help="Save JSON results to file")
    run_parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    run_parser.add_argument("--iterations", type=int, default=None, help="Measurement iterations (default: auto)")

    # compare
    cmp_parser = subparsers.add_parser("compare", help="Compare two benchmark results")
    cmp_parser.add_argument("baseline", help="Baseline JSON results file")
    cmp_parser.add_argument("current", help="Current JSON results file")

    args = parser.parse_args()

    match args.command:
        case "run":
            _cmd_run(args)
        case "compare":
            _cmd_compare(args)
