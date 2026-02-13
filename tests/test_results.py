from pybench.results import BenchmarkResult


def test_from_times_computes_statistics():
    times_ns = [100_000, 200_000, 300_000, 400_000, 500_000]
    result = BenchmarkResult.from_times("test_bench", times_ns)

    assert result.name == "test_bench"
    assert result.iterations == 5
    assert result.times_ns == times_ns
    assert result.min_ns == 100_000
    assert result.max_ns == 500_000
    assert result.mean_ns == 300_000.0
    assert result.median_ns == 300_000.0
    assert result.ops_per_sec > 0


def test_from_times_stddev():
    # All same value => stddev 0
    times_ns = [100_000, 100_000, 100_000]
    result = BenchmarkResult.from_times("steady", times_ns)
    assert result.stddev_ns == 0.0


def test_from_times_single_sample():
    times_ns = [500_000]
    result = BenchmarkResult.from_times("single", times_ns)
    assert result.mean_ns == 500_000.0
    assert result.median_ns == 500_000.0
    assert result.stddev_ns == 0.0


def test_to_dict_roundtrip():
    times_ns = [100_000, 200_000, 300_000]
    result = BenchmarkResult.from_times("roundtrip", times_ns)
    d = result.to_dict()

    assert d["name"] == "roundtrip"
    assert d["iterations"] == 3
    assert d["mean_ns"] == result.mean_ns
    assert d["median_ns"] == result.median_ns
    assert d["stddev_ns"] == result.stddev_ns
    assert d["min_ns"] == 100_000
    assert d["max_ns"] == 300_000
    assert "ops_per_sec" in d
