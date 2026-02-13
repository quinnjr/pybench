from pybench.bench import Bench
from pybench.results import BenchmarkResult


def test_decorator_registers_function():
    bench = Bench()

    @bench.benchmark
    def my_fn():
        sum(range(100))

    assert len(bench._registered) == 1
    assert bench._registered[0][0] == "my_fn"


def test_decorator_with_options():
    bench = Bench()

    @bench.benchmark(warmup=2, iterations=10)
    def my_fn():
        sum(range(100))

    assert len(bench._registered) == 1
    name, fn, opts = bench._registered[0]
    assert opts["warmup"] == 2
    assert opts["iterations"] == 10


def test_context_manager_records_timing():
    bench = Bench(warmup=0, iterations=1)

    with bench.measure("inline"):
        sum(range(100))

    assert len(bench._results) == 1
    assert bench._results[0].name == "inline"


def test_run_executes_registered_benchmarks():
    bench = Bench(warmup=0, iterations=3)

    @bench.benchmark
    def add_numbers():
        1 + 1

    results = bench.run()

    assert len(results) == 1
    assert results[0].name == "add_numbers"
    assert results[0].iterations == 3


def test_run_includes_context_manager_results():
    bench = Bench(warmup=0, iterations=3)

    with bench.measure("ctx"):
        1 + 1

    @bench.benchmark
    def deco():
        1 + 1

    results = bench.run()

    names = [r.name for r in results]
    assert "ctx" in names
    assert "deco" in names


def test_module_level_decorator():
    """Test the module-level benchmark decorator that uses a global registry."""
    from pybench.bench import benchmark

    @benchmark
    def standalone():
        sum(range(10))

    # Should be callable — decorator doesn't break the function
    standalone()


def test_module_level_decorator_with_options():
    """Test module-level benchmark decorator with warmup/iterations kwargs."""
    from pybench.bench import benchmark, _global_registry

    before = len(_global_registry)

    @benchmark(warmup=2, iterations=10)
    def configured():
        sum(range(10))

    assert len(_global_registry) == before + 1
    name, fn, opts = _global_registry[-1]
    assert name == "configured"
    assert opts["warmup"] == 2
    assert opts["iterations"] == 10
    configured()


def test_report_auto_runs(capsys):
    """report() should auto-run benchmarks if not yet executed."""
    bench = Bench(warmup=0, iterations=3)

    @bench.benchmark
    def auto_run():
        1 + 1

    bench.report()

    captured = capsys.readouterr()
    assert "auto_run" in captured.out


def test_report_json_output(capsys):
    """report(json_output=True) should print JSON."""
    import json

    bench = Bench(warmup=0, iterations=3)

    @bench.benchmark
    def json_test():
        1 + 1

    bench.run()
    bench.report(json_output=True)

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "results" in data


def test_to_json_auto_runs():
    """to_json() should auto-run benchmarks if not yet executed."""
    import json

    bench = Bench(warmup=0, iterations=3)

    @bench.benchmark
    def auto_json():
        1 + 1

    raw = bench.to_json()
    data = json.loads(raw)
    assert any(r["name"] == "auto_json" for r in data["results"])
