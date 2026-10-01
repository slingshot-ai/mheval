"""Summarize a results directory as Markdown tables (one row per model, one column per task).
The best score in each column is bolded (ties included), using each metric's own direction.

    python -m mheval.report results/
    python -m mheval.report results/ --models gpt-5-2025-08-07 anthropic__claude-sonnet-4.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_tasks


def collect(results_dir: Path) -> dict[str, dict[str, dict]]:
    """{model_dir: {task: results.json}} for every finished task."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(results_dir.glob("*/*/results.json")):
        out.setdefault(path.parent.parent.name, {})[path.parent.name] = json.loads(path.read_text())
    return out


def fmt(value) -> str:
    if value is None:
        return "–"
    return f"{value:.2f}" if abs(value) >= 10 else f"{value:.3f}"


def tables(results_dir: Path, models: list[str] | None = None) -> str:
    tasks = load_tasks()
    runs = collect(results_dir)
    if models:
        runs = {m: runs[m] for m in models}
    sections = []
    for tag in ("quality", "safety"):
        cols = [t for t in sorted(tasks.values(), key=lambda c: c.task) if tag in t.tag]
        head, columns = ["Model"], []
        for c in cols:
            primary = c.metric_list[0]
            higher = primary.get("higher_is_better", True)
            head.append(f"{c.task}<br>`{primary['metric']}` {'↑' if higher else '↓'}")
            values = [res.get(c.task, {}).get("metrics", {}).get(primary["metric"]) for res in runs.values()]
            shown = [fmt(v) for v in values]
            scored = [(v, s) for v, s in zip(values, shown) if v is not None]
            best = (max if higher else min)(scored)[1] if scored else None
            columns.append([f"**{s}**" if s == best else s for s in shown])
        rows = [head, ["---"] + ["---:"] * len(cols)]
        for i, model in enumerate(runs):
            rows.append([f"`{model.replace('__', '/')}`", *(col[i] for col in columns)])
        sections.append(f"### {tag.capitalize()}\n\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows))
    return "\n\n".join(sections)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path, nargs="?", default=Path("results"))
    p.add_argument("--models", nargs="+", help="Model result directories to include (default: all).")
    a = p.parse_args()
    print(tables(a.results_dir, a.models))


if __name__ == "__main__":
    main()
