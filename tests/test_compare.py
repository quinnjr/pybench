import json

from pybench.compare import compare_results, format_comparison


def _make_json(results: list[dict]) -> str:
    return json.dumps({"metadata": {}, "results": results})


def test_compare_results_matching_benchmarks():
    baseline = _make_json([
        {"name": "sort", "mean_ns": 1000.0, "median_ns": 1000.0, "stddev_ns": 10.0,
         "min_ns": 990, "max_ns": 1010, "iterations": 100, "ops_per_sec": 1_000_000.0},
    ])
    current = _make_json([
        {"name": "sort", "mean_ns": 1200.0, "median_ns": 1200.0, "stddev_ns": 12.0,
         "min_ns": 1190, "max_ns": 1210, "iterations": 100, "ops_per_sec": 833_333.0},
    ])

    diffs = compare_results(baseline, current)

    assert len(diffs) == 1
    assert diffs[0]["name"] == "sort"
    assert diffs[0]["baseline_mean_ns"] == 1000.0
    assert diffs[0]["current_mean_ns"] == 1200.0
    assert diffs[0]["change_pct"] == 20.0  # 20% slower


def test_compare_results_improvement():
    baseline = _make_json([
        {"name": "hash", "mean_ns": 2000.0, "median_ns": 2000.0, "stddev_ns": 20.0,
         "min_ns": 1990, "max_ns": 2010, "iterations": 50, "ops_per_sec": 500_000.0},
    ])
    current = _make_json([
        {"name": "hash", "mean_ns": 1000.0, "median_ns": 1000.0, "stddev_ns": 10.0,
         "min_ns": 990, "max_ns": 1010, "iterations": 50, "ops_per_sec": 1_000_000.0},
    ])

    diffs = compare_results(baseline, current)

    assert diffs[0]["change_pct"] == -50.0  # 50% faster


def test_compare_results_missing_in_current():
    baseline = _make_json([
        {"name": "gone", "mean_ns": 1000.0, "median_ns": 1000.0, "stddev_ns": 10.0,
         "min_ns": 990, "max_ns": 1010, "iterations": 50, "ops_per_sec": 1_000_000.0},
    ])
    current = _make_json([])

    diffs = compare_results(baseline, current)

    assert len(diffs) == 1
    assert diffs[0]["name"] == "gone"
    assert diffs[0]["current_mean_ns"] is None
    assert diffs[0]["change_pct"] is None


def test_format_comparison_contains_names():
    baseline = _make_json([
        {"name": "sort", "mean_ns": 1000.0, "median_ns": 1000.0, "stddev_ns": 10.0,
         "min_ns": 990, "max_ns": 1010, "iterations": 100, "ops_per_sec": 1_000_000.0},
    ])
    current = _make_json([
        {"name": "sort", "mean_ns": 1100.0, "median_ns": 1100.0, "stddev_ns": 11.0,
         "min_ns": 1090, "max_ns": 1110, "iterations": 100, "ops_per_sec": 909_091.0},
    ])

    output = format_comparison(baseline, current)
    assert "sort" in output
    assert "%" in output
