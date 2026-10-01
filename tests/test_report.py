import json

from mheval.report import tables


def write(root, model, task, metrics):
    path = root / model / task / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"metrics": metrics, "breakdown": {}, "n": 1}))


def test_tables_bold_best_per_direction_and_filter(tmp_path):
    write(tmp_path, "a", "mindeval", {"average_score": 3.5})
    write(tmp_path, "b", "mindeval", {"average_score": 3.9})
    write(tmp_path, "a", "sim_vail", {"mh_harm_mean": 1.2})  # lower is better
    write(tmp_path, "b", "sim_vail", {"mh_harm_mean": 2.4})
    write(tmp_path, "c", "mindeval", {"average_score": 9.9})

    out = tables(tmp_path, models=["a", "b"])
    assert "| `b` |" in out and "`c`" not in out
    assert "**3.900**" in out and "**3.500**" not in out
    assert "**1.200**" in out and "**2.400**" not in out


def test_tables_bold_ties(tmp_path):
    write(tmp_path, "a", "counselbench_eval", {"overall": 5.0})
    write(tmp_path, "b", "counselbench_eval", {"overall": 5.0})
    assert tables(tmp_path).count("**5.000**") == 2
