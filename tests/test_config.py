import dataclasses

import pytest

from mheval.config import Role, TaskConfig, load_tasks, parse_kv, select


def test_all_tasks_load_with_the_same_shape():
    tasks = load_tasks()
    assert len(tasks) == 10
    keys = {f.name for f in dataclasses.fields(TaskConfig)} - {"task_dir"}
    for cfg in tasks.values():
        assert set(vars(cfg)) - {"task_dir"} == keys
        assert cfg.tag in (["quality"], ["safety"])
        assert cfg.metric_list and all("metric" in m for m in cfg.metric_list)
        assert {"version", "description", "paper", "homepage", "implementation"} <= cfg.metadata.keys()
        for src in cfg.source:
            assert ("repo" in src and len(src["commit"]) == 40) or ("url" in src and len(src["sha256"]) == 64)


def test_select_by_name_and_tag():
    tasks = load_tasks()
    assert [t.task for t in select(tasks, ["mindeval"])] == ["mindeval"]
    assert {t.task for t in select(tasks, ["safety"])} == {"vera_mh", "spiral_bench", "sim_vail"}
    with pytest.raises(SystemExit):
        select(tasks, ["nope"])


def test_parse_kv():
    assert parse_kv("model=a/b,c/d,base_url=http://x/v1,temperature=0.5,flag=true,empty=") == {
        "model": "a/b,c/d", "base_url": "http://x/v1", "temperature": 0.5, "flag": True, "empty": ""}
    assert parse_kv("model=1e3") == {"model": "1e3"}  # model ids stay strings
    assert parse_kv(None) == {}


def test_role_override_rules():
    default = {"model": "m", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
               "reasoning_effort": "high"}
    # same provider, new model: keep endpoint + key, drop model-specific params
    r = Role.parse("model=other", default)
    assert (r.base_url, r.api_key_env, r.params) == ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", {})
    # new endpoint without a key: never forward the default provider's key
    r = Role.parse("base_url=http://localhost:8000/v1", default)
    assert r.api_key_env == "" and r.api_key == "EMPTY"
    # extra keys become request params
    assert Role.parse("model=x,temperature=0").params == {"temperature": 0}
    assert Role.parse(None, None) is None
