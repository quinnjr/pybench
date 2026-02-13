"""End-to-end integration test exercising the full pybench workflow."""

import json


def test_full_workflow():
    from pybench import Bench

    bench = Bench(warmup=1, iterations=10)

    # Decorator API
    @bench.benchmark
    def list_sort():
        sorted([3, 1, 4, 1, 5, 9, 2, 6])

    @bench.benchmark(warmup=0, iterations=5)
    def string_concat():
        "".join(["hello"] * 100)

    # Context manager API
    with bench.measure("dict_creation"):
        {i: i * 2 for i in range(100)}

    # Run all
    results = bench.run()

    assert len(results) == 3
    names = [r.name for r in results]
    assert "list_sort" in names
    assert "string_concat" in names
    assert "dict_creation" in names

    # JSON roundtrip
    raw = bench.to_json()
    data = json.loads(raw)
    assert len(data["results"]) == 3
    assert "metadata" in data
