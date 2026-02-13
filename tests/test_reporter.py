import json
import platform
import sys

from pybench.reporter import format_table, format_json, format_time
from pybench.results import BenchmarkResult


def _make_result(name: str, times_ns: list[int]) -> BenchmarkResult:
    return BenchmarkResult.from_times(name, times_ns)


def test_format_time_nanoseconds():
    assert format_time(500) == "500.0 ns"


def test_format_time_microseconds():
    assert format_time(1_500) == "1.5 µs"


def test_format_time_milliseconds():
    assert format_time(1_500_000) == "1.5 ms"


def test_format_time_seconds():
    assert format_time(1_500_000_000) == "1.50 s"


def test_format_table_contains_header_and_names():
    results = [
        _make_result("fast", [1_000, 1_100, 1_200]),
        _make_result("slow", [100_000, 110_000, 120_000]),
    ]
    output = format_table(results)

    assert "Name" in output
    assert "Mean" in output
    assert "Ops/sec" in output
    assert "fast" in output
    assert "slow" in output


def test_format_json_structure():
    results = [_make_result("bench1", [50_000, 60_000])]
    raw = format_json(results)
    data = json.loads(raw)

    assert "metadata" in data
    assert "results" in data
    assert data["metadata"]["python_version"] == platform.python_version()
    assert data["metadata"]["platform"] == platform.system()
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "bench1"


def test_format_json_is_valid_json():
    results = [_make_result("x", [1000])]
    raw = format_json(results)
    # Should not raise
    json.loads(raw)
