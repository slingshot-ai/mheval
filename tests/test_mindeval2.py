import json

import pytest

from mheval.config import load_tasks

WEIGHTS = {"No concern": 0, "Minor": 1, "Major": 5, "Critical": 10}


def write_run(out, verdicts, sessions_expected):
    """judge/mindeval2/ as mind-eval writes it: verdicts.jsonl, and summary.json with the fields the task reads."""
    judge_dir = out / "judge" / "mindeval2"
    judge_dir.mkdir(parents=True)
    (judge_dir / "verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in verdicts))
    sessions = {(v["member"], v["episode"]) for v in verdicts}
    summary = {
        "n_replies": len(verdicts), "errors": 0, "replies": {"pending": 0},
        "coverage": {"sessions_judged": len(sessions), "sessions_expected": sessions_expected},
        "mqm": {"weights": WEIGHTS, "sessions": len(sessions), "mean": 2.0, "p99": 3.0,
                "incidents": {"critical": 0.0, "major_or_worse": 0.5}},
        "severity_rates": {"No concern": 0.5, "Minor": 0.25, "Major": 0.25, "Critical": 0.0},
        "issue_rates": {"over_exploring": 0.25},
    }
    (judge_dir / "summary.json").write_text(json.dumps(summary))


def process_results():
    cfg = load_tasks()["mindeval2"]
    return cfg.process_results.resolve(cfg.task_dir)


def test_when_every_session_is_judged_then_mhqr_is_the_summary_mean_and_sessions_are_broken_down(tmp_path):
    verdicts = [
        {"member": "m1", "episode": "ep001", "severity": "No concern"},
        {"member": "m1", "episode": "ep001", "severity": "Major"},
        {"member": "m1", "episode": "ep002", "severity": "Minor"},
        {"member": "m1", "episode": "ep002", "severity": "No concern"},
    ]
    write_run(tmp_path, verdicts, sessions_expected=2)

    res = process_results()(tmp_path, {})

    assert res["metrics"]["mhqr"] == 2.0
    assert res["breakdown"]["session"] == {"ep001": {"mhqr": 2.5, "n": 1}, "ep002": {"mhqr": 0.5, "n": 1}}
    assert res["breakdown"]["severity"]["no_concern"] == {"rate": 0.5}
    assert res["n"] == 2


def test_when_a_session_is_not_judged_then_the_run_fails(tmp_path):
    write_run(tmp_path, [{"member": "m1", "episode": "ep001", "severity": "Minor"}], sessions_expected=3)

    with pytest.raises(RuntimeError, match="incomplete run"):
        process_results()(tmp_path, {})
