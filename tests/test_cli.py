import json
import os
import textwrap
import tempfile
from pathlib import Path
from unittest.mock import patch

from pybench.cli import main


def _create_bench_file(tmp_path: Path, content: str) -> Path:
    f = tmp_path / "bench_example.py"
    f.write_text(content)
    return f


def test_cli_run_discovers_bench_files(tmp_path):
    _create_bench_file(tmp_path, textwrap.dedent("""\
        import pybench

        @pybench.benchmark
        def bench_add():
            1 + 1
    """))

    with patch("sys.argv", ["pybench", "run", str(tmp_path), "--iterations", "3", "--warmup", "0"]):
        # Should not raise
        main()


def test_cli_run_json_output(tmp_path, capsys):
    _create_bench_file(tmp_path, textwrap.dedent("""\
        import pybench

        @pybench.benchmark
        def bench_add():
            1 + 1
    """))

    with patch("sys.argv", ["pybench", "run", str(tmp_path), "--json", "--iterations", "3", "--warmup", "0"]):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "results" in data
    assert len(data["results"]) >= 1


def test_cli_run_save_json(tmp_path):
    _create_bench_file(tmp_path, textwrap.dedent("""\
        import pybench

        @pybench.benchmark
        def bench_add():
            1 + 1
    """))

    out_file = tmp_path / "results.json"
    with patch("sys.argv", ["pybench", "run", str(tmp_path), "--save", str(out_file), "--iterations", "3", "--warmup", "0"]):
        main()

    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "results" in data


def test_cli_run_single_file(tmp_path, capsys):
    bench_file = _create_bench_file(tmp_path, textwrap.dedent("""\
        import pybench

        @pybench.benchmark
        def bench_add():
            1 + 1
    """))

    with patch("sys.argv", ["pybench", "run", str(bench_file), "--json", "--iterations", "3", "--warmup", "0"]):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert len(data["results"]) >= 1


def test_cli_run_no_benchmarks(tmp_path):
    with patch("sys.argv", ["pybench", "run", str(tmp_path), "--iterations", "3", "--warmup", "0"]):
        import pytest
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_cli_compare(tmp_path, capsys):
    baseline = {"metadata": {}, "results": [
        {"name": "sort", "mean_ns": 1000.0, "median_ns": 1000.0, "stddev_ns": 10.0,
         "min_ns": 990, "max_ns": 1010, "iterations": 100, "ops_per_sec": 1_000_000.0},
    ]}
    current = {"metadata": {}, "results": [
        {"name": "sort", "mean_ns": 1100.0, "median_ns": 1100.0, "stddev_ns": 11.0,
         "min_ns": 1090, "max_ns": 1110, "iterations": 100, "ops_per_sec": 909_091.0},
    ]}

    base_file = tmp_path / "baseline.json"
    curr_file = tmp_path / "current.json"
    base_file.write_text(json.dumps(baseline))
    curr_file.write_text(json.dumps(current))

    with patch("sys.argv", ["pybench", "compare", str(base_file), str(curr_file)]):
        main()

    captured = capsys.readouterr()
    assert "sort" in captured.out
    assert "%" in captured.out
