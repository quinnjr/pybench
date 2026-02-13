from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

from pybench.results import BenchmarkResult


def format_time(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.1f} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.1f} \u00b5s"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def format_table(results: list[BenchmarkResult]) -> str:
    if not results:
        return "No benchmark results."

    headers = ["Name", "Mean", "Median", "StdDev", "Min", "Max", "Ops/sec"]

    rows: list[list[str]] = []
    for r in results:
        rows.append([
            r.name,
            format_time(r.mean_ns),
            format_time(r.median_ns),
            format_time(r.stddev_ns),
            format_time(r.min_ns),
            format_time(r.max_ns),
            f"{r.ops_per_sec:,.0f}",
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i == 0:
                parts.append(cell.ljust(col_widths[i]))
            else:
                parts.append(cell.rjust(col_widths[i]))
        return "  ".join(parts)

    sep = "\u2500" * (sum(col_widths) + 2 * (len(headers) - 1))
    lines = [
        "pybench results",
        sep,
        fmt_row(headers),
        sep,
    ]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def format_json(results: list[BenchmarkResult]) -> str:
    data = {
        "metadata": {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": [r.to_dict() for r in results],
    }
    return json.dumps(data, indent=2)
