from __future__ import annotations

import json

from pybench.reporter import format_time


def compare_results(baseline_json: str, current_json: str) -> list[dict]:
    baseline_data = json.loads(baseline_json)
    current_data = json.loads(current_json)

    current_by_name = {r["name"]: r for r in current_data["results"]}
    diffs = []

    for b in baseline_data["results"]:
        name = b["name"]
        c = current_by_name.pop(name, None)

        if c is None:
            diffs.append({
                "name": name,
                "baseline_mean_ns": b["mean_ns"],
                "current_mean_ns": None,
                "change_pct": None,
            })
        else:
            change = ((c["mean_ns"] - b["mean_ns"]) / b["mean_ns"]) * 100
            diffs.append({
                "name": name,
                "baseline_mean_ns": b["mean_ns"],
                "current_mean_ns": c["mean_ns"],
                "change_pct": round(change, 1),
            })

    # New benchmarks not in baseline
    for name, c in current_by_name.items():
        diffs.append({
            "name": name,
            "baseline_mean_ns": None,
            "current_mean_ns": c["mean_ns"],
            "change_pct": None,
        })

    return diffs


def format_comparison(baseline_json: str, current_json: str) -> str:
    diffs = compare_results(baseline_json, current_json)

    if not diffs:
        return "No benchmarks to compare."

    headers = ["Name", "Baseline", "Current", "Change"]

    rows: list[list[str]] = []
    for d in diffs:
        baseline_str = format_time(d["baseline_mean_ns"]) if d["baseline_mean_ns"] is not None else "N/A"
        current_str = format_time(d["current_mean_ns"]) if d["current_mean_ns"] is not None else "N/A"

        if d["change_pct"] is None:
            change_str = "N/A"
        elif d["change_pct"] > 0:
            change_str = f"+{d['change_pct']:.1f}% (slower)"
        elif d["change_pct"] < 0:
            change_str = f"{d['change_pct']:.1f}% (faster)"
        else:
            change_str = "0.0% (same)"

        rows.append([d["name"], baseline_str, current_str, change_str])

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
        "pybench comparison",
        sep,
        fmt_row(headers),
        sep,
    ]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)
