"""Summarize an eqbench3.py run into results.json, reusing `core.benchmark.calculate_final_rubric_score`
on per-scenario-type slices of the run for the breakdown."""
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
import utils.constants as C  # noqa: E402
from core.benchmark import calculate_final_rubric_score  # noqa: E402

from mheval.native import write_results  # noqa: E402


def scenario_type(sid):
    if sid in C.ANALYSIS_SCENARIO_IDS:
        return "analysis"
    return "message_drafting" if sid in C.MESSAGE_DRAFTING_SCENARIO_IDS else "roleplay"


def main(runs_file, elo_file, model_name):
    runs = json.loads(Path(runs_file).read_text())
    run = next(r for k, r in runs.items() if not k.startswith("__") and r.get("model_name") == model_name)
    res = run.get("results", {})
    metrics = {"rubric_score": res["average_rubric_score"] * 5}  # 0-20 -> 0-100, as printed by eqbench3.py
    if isinstance(res.get("elo_normalized"), (int, float)):
        metrics |= {"elo_normalized": res["elo_normalized"], "elo_raw": res["elo_raw"]}
        elo = json.loads(Path(elo_file).read_text()).get(model_name, {})
        metrics |= {k: elo[k] for k in ("ci_low_norm", "ci_high_norm") if k in elo}

    by_type, criteria, n = defaultdict(lambda: defaultdict(dict)), defaultdict(list), 0
    for it, tasks in run["scenario_tasks"].items():
        for sid, task in tasks.items():
            if task.get("status") == "rubric_scored" and task.get("rubric_scores"):
                by_type[scenario_type(sid)][it][sid] = task
                n += 1
                for c, v in task["rubric_scores"].items():
                    if isinstance(v, (int, float)):
                        criteria[c].append(v)
    breakdown = {
        "scenario_type": {t: {"rubric_score": calculate_final_rubric_score({"scenario_tasks": tasks})[0] * 5,
                              "n": sum(map(len, tasks.values()))} for t, tasks in sorted(by_type.items())},
        "criterion": {c: {"mean_0_20": statistics.mean(v), "n": len(v)} for c, v in sorted(criteria.items())},
    }
    write_results(metrics, breakdown, n)


if __name__ == "__main__":
    main(*sys.argv[1:])
