"""Command-line entry point.

    mheval --tasks list
    mheval --tasks quality --model_args model=Qwen/Qwen3-8B,base_url=http://localhost:8000/v1
    mheval --tasks vera_mh,spiral_bench --model_args model=gpt-5.2 --judge_args model=gpt-5.4 --limit 5
"""
from __future__ import annotations

import argparse
import datetime
import json
import traceback
from pathlib import Path

from . import submission
from .config import load_tasks, parse_kv, select
from .evaluator import run_task
from .workspace import prepare


def main() -> None:
    p = argparse.ArgumentParser(prog="mheval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", required=True, help="Comma-separated task names or tags (quality, safety), or `list`.")
    p.add_argument("--model_args", help="Model under test: model=<id>[,base_url=<…/v1>][,api_key_env=<VAR>].")
    p.add_argument("--judge_args", help="Override the task's default judge (same syntax).")
    p.add_argument("--user_args", help="Override the task's default user simulator (same syntax).")
    p.add_argument("--gen_kwargs", help="Target sampling overrides, e.g. temperature=0,max_tokens=2048.")
    p.add_argument("--task_args", help="Benchmark-specific overrides, e.g. n_turns=10 (see each task YAML).")
    p.add_argument("--limit", type=int, help="Evaluate only the first N items (personas, prompts, ...).")
    p.add_argument("--output_path", default="results", type=Path)
    p.add_argument("--include_path", action="append", default=[], help="Extra directory of task YAMLs.")
    p.add_argument("--setup_only", action="store_true", help="Only fetch sources and build task envs.")
    a = p.parse_args()

    all_tasks = load_tasks(a.include_path)
    if a.tasks == "list":
        for cfg in sorted(all_tasks.values(), key=lambda c: (c.tag, c.task)):
            print(f"{cfg.task:28} {','.join(cfg.tag):22} {cfg.metadata.get('description', '')}")
        return
    tasks = select(all_tasks, a.tasks.split(","))
    if a.setup_only:
        for cfg in tasks:
            print(f"{cfg.task}: {prepare(cfg)}")
        return
    if not a.model_args:
        p.error("--model_args is required")

    roles = {"target": a.model_args, "judge": a.judge_args, "user": a.user_args}
    model = parse_kv(a.model_args)["model"]
    run_dir = a.output_path / submission.model_id(str(model))
    results, failed = {}, {}
    for cfg in tasks:
        try:
            results[cfg.task] = run_task(cfg, roles, run_dir / cfg.task, limit=a.limit,
                                         gen_kwargs=parse_kv(a.gen_kwargs), task_args=parse_kv(a.task_args))
            if a.limit is None:  # only full runs are eligible for the leaderboard
                submission.write(run_dir, str(model), cfg.task,
                                 json.loads((run_dir / cfg.task / "results.json").read_text()))
        except Exception as e:  # a failed task is reported at the end; the others still run
            traceback.print_exc()
            failed[cfg.task] = repr(e)

    out = {"results": results, "failed": failed,
           "config": {k: v for k, v in vars(a).items() if k != "output_path"},
           "configs": {c.task: {"source": c.source, "metadata": c.metadata} for c in tasks},
           "date": datetime.datetime.now().isoformat(timespec="seconds")}
    path = run_dir / f"results_{datetime.datetime.now():%Y-%m-%dT%H-%M-%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(table(results, {c.task: c for c in tasks}))
    for task, err in failed.items():
        print(f"FAILED {task}: {err}")
    print(f"Saved {path}")
    if a.limit is None and results:
        print(f"Leaderboard submission files: {run_dir / 'submission'} (see the leaderboard README)")


def table(results: dict, cfgs: dict) -> str:
    rows = ["|Task|Tag|Metric|↑↓|Value|n|", "|---|---|---|---|---:|---:|"]
    for task, r in results.items():
        for m in cfgs[task].metric_list:
            v = r["metrics"].get(m["metric"])
            arrow = "↑" if m.get("higher_is_better", True) else "↓"
            val = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            rows.append(f"|{task}|{','.join(cfgs[task].tag)}|{m['metric']}|{arrow}|{val}|{r.get('n') or ''}|")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
