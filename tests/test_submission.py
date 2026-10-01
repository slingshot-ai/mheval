import json

from mheval import submission
from mheval.config import load_tasks, task_sha256


def test_model_id():
    assert submission.model_id("anthropic/claude-sonnet-4.5") == "anthropic__claude-sonnet-4.5"
    assert submission.model_id("gpt-5-2025-08-07") == "gpt-5-2025-08-07"


def test_write_lays_out_leaderboard_files(tmp_path):
    result = {"metrics": {"average_score": 3.5}, "breakdown": {}, "n": 50, "meta": {"task": "mindeval"}}
    out = submission.write(tmp_path, "org/model-x", "mindeval", result)
    assert out == tmp_path / "submission/results/org__model-x/mindeval.json"
    assert json.loads(out.read_text()) == result
    card = tmp_path / "submission/models/org__model-x.yaml"
    assert "name: org/model-x" in card.read_text()

    card.write_text("edited")  # a filled-in model card is never overwritten by later tasks
    submission.write(tmp_path, "org/model-x", "cbt_bench", result)
    assert card.read_text() == "edited"
    assert (tmp_path / "submission/results/org__model-x/cbt_bench.json").exists()


def test_task_sha256_is_stable_and_distinct():
    tasks = load_tasks()
    hashes = {name: task_sha256(cfg) for name, cfg in tasks.items()}
    assert hashes == {name: task_sha256(cfg) for name, cfg in tasks.items()}
    assert len(set(hashes.values())) == len(hashes) - 1  # counselbench_eval/_adv share one directory
