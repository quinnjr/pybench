def test_public_api_imports():
    from pybench import Bench, BenchmarkResult, benchmark

    assert Bench is not None
    assert BenchmarkResult is not None
    assert callable(benchmark)


def test_bench_report(capsys):
    from pybench import Bench

    bench = Bench(warmup=0, iterations=3)

    @bench.benchmark
    def trivial():
        1 + 1

    bench.run()
    bench.report()

    captured = capsys.readouterr()
    assert "trivial" in captured.out


def test_bench_to_json():
    import json
    from pybench import Bench

    bench = Bench(warmup=0, iterations=3)

    @bench.benchmark
    def trivial():
        1 + 1

    bench.run()
    raw = bench.to_json()
    data = json.loads(raw)
    assert "results" in data
