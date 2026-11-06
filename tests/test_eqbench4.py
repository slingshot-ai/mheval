import importlib.util
import json

import pytest

from mheval.config import load_tasks

TARGET = "vendor/new-model"


def task():
    return load_tasks()["eqbench4"]


def prepare_module():
    spec = importlib.util.spec_from_file_location("eqbench4_prepare", task().task_dir / "prepare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_file(model, conversations, pairwise=()):
    run = {"__meta__": {"evaluated_model": model}, "__pairwise__": list(pairwise)}
    run.update({f"s{i}": [c] for i, c in enumerate(conversations)})
    return {"r1": run}


def comparison(a, b, fraction_for_a):
    return {"scenario_id": "s0", "iteration_index": 0, "pair": {"model_a": a, "model_b": b},
            "aggregated": {"fraction_for_model_a": fraction_for_a}}


def write_output(out, completed, rated=True):
    results = out / "results"
    results.mkdir(parents=True)
    traits = {"challenge_propensity": 4.0, "num_judges": 1}
    convos = [{"completed": True, "performance_judge_results": traits}] * completed
    pairwise = [comparison(TARGET, "pool/a", 0.75), comparison("pool/b", TARGET, 0.75)]
    (results / "mheval.json").write_text(json.dumps(run_file(TARGET, convos, pairwise)))
    ratings = {TARGET: {"elo_norm": 1100.0, "ci_low_norm": 1090.0, "ci_high_norm": 1110.0}} if rated else {}
    (results / "elo_results.json").write_text(json.dumps({"model_ratings": ratings}))


@pytest.fixture
def ctx():
    return {"target": {"model": TARGET}, "limit": 2}


def test_when_every_conversation_finished_and_was_rated_then_elo_and_win_fraction_are_reported(tmp_path, ctx):
    write_output(tmp_path, completed=2)

    res = task().process_results.resolve(task().task_dir)(tmp_path, ctx)

    assert res["metrics"] == {"elo": 1100.0, "elo_ci_low": 1090.0, "elo_ci_high": 1110.0, "pairwise_win_fraction": 0.5}
    assert res["breakdown"]["trait"] == {"challenge_propensity": {"score": 4.0}}


def test_when_a_conversation_is_unfinished_then_the_run_fails(tmp_path, ctx):
    write_output(tmp_path, completed=1)

    with pytest.raises(RuntimeError, match="incomplete run"):
        task().process_results.resolve(task().task_dir)(tmp_path, ctx)


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """A bundled pool of two models, the second compared against the first."""
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    (repo / "results" / "r1.json").write_text(json.dumps(run_file("pool/a", [])))
    (repo / "results" / "r2.json").write_text(json.dumps(run_file("pool/b", [], [comparison("pool/a", "pool/b", 0.5)])))
    ratings = {"pool/a": {"elo_norm": 1000.0}, "pool/b": {"elo_norm": 1200.0}}
    (repo / "results" / "elo_results.json").write_text(json.dumps({"model_ratings": ratings}))
    monkeypatch.chdir(repo)
    return tmp_path / "out"


def test_when_the_model_is_new_then_the_run_links_the_whole_pool(pool):
    prepare_module().link_pool(pool, TARGET, None)

    assert sorted(p.name for p in (pool / "results").iterdir()) == ["elo_results.json", "r1.json", "r2.json"]
    assert (pool / "results" / "r1.json").is_symlink()


def test_when_the_model_is_already_in_the_pool_then_the_run_is_refused(pool):
    with pytest.raises(SystemExit, match="already in the EQ-Bench 4 pool"):
        prepare_module().link_pool(pool, "pool/a", None)


def test_when_a_pool_model_is_held_out_then_its_run_and_comparisons_are_left_out(pool):
    prepare_module().link_pool(pool, "pool/a", "pool/a")

    assert not (pool / "results" / "r1.json").exists()
    assert json.loads((pool / "results" / "r2.json").read_text())["r1"]["__pairwise__"] == []
    assert list(json.loads((pool / "results" / "elo_results.json").read_text())["model_ratings"]) == ["pool/b"]
