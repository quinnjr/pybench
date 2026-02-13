from pybench.runner import Runner
from pybench.results import BenchmarkResult


def test_run_function_returns_result():
    def noop():
        pass

    runner = Runner(warmup=1, iterations=5)
    result = runner.run("noop", noop)

    assert isinstance(result, BenchmarkResult)
    assert result.name == "noop"
    assert result.iterations == 5
    assert len(result.times_ns) == 5
    assert all(t > 0 for t in result.times_ns)


def test_run_function_with_warmup():
    call_count = 0

    def counting():
        nonlocal call_count
        call_count += 1

    runner = Runner(warmup=3, iterations=2)
    runner.run("counting", counting)

    # 3 warmup + 2 measured = 5 total calls
    assert call_count == 5


def test_auto_calibrate():
    runner = Runner(warmup=0, iterations=None, target_time_ns=10_000_000)

    def fast_fn():
        sum(range(10))

    result = runner.run("fast", fast_fn)
    # auto-calibration should pick more than 1 iteration
    assert result.iterations >= 1
    assert len(result.times_ns) == result.iterations


def test_run_respects_gc_disable():
    """GC should be disabled during measurement but restored after."""
    import gc

    gc_was_enabled = gc.isenabled()
    runner = Runner(warmup=0, iterations=3)

    gc_states_during = []

    def check_gc():
        gc_states_during.append(gc.isenabled())

    runner.run("gc_check", check_gc)

    # GC should be disabled during runs
    assert not any(gc_states_during)
    # GC should be restored to original state after
    assert gc.isenabled() == gc_was_enabled
