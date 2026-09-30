"""Summarize a results directory as Markdown tables (one row per model, one column per task).

    python -m mheval.report results/
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


def tables(results_dir: Path) -> str:
    tasks = load_tasks()
    runs = collect(results_dir)
    sections = []
    for tag in ("quality", "safety"):
        cols = [t for t in sorted(tasks.values(), key=lambda c: c.task) if tag in t.tag]
        head = ["Model"]
        for c in cols:
            primary = c.metric_list[0]
            arrow = "↑" if primary.get("higher_is_better", True) else "↓"
            head.append(f"{c.task}<br>`{primary['metric']}` {arrow}")
        rows = [head, ["---"] + ["---:"] * len(cols)]
        for model, res in runs.items():
            cells = [fmt(res.get(c.task, {}).get("metrics", {}).get(c.metric_list[0]["metric"])) for c in cols]
            rows.append([f"`{model.replace('__', '/')}`", *cells])
        sections.append(f"#### {tag.capitalize()}\n\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows))
    return "\n\n".join(sections)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path, nargs="?", default=Path("results"))
    print(tables(p.parse_args().results_dir))


if __name__ == "__main__":
    main()
