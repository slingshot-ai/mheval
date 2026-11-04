import json
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mheval.config import load_tasks

# Petri's TranscriptHook.on_sample_end cut down to the two lines that name the transcript file, which are verbatim
# from src/petri/hooks/transcript_hook.py at the commit sim_vail pins (MIT, Copyright (c) 2025 Safety Research).
PETRI_HOOK = '''\
def save(sample, save_dir, current_time, transcript):
    with open(os.path.join(save_dir, f"transcript_{current_time}_{sample.id}.json"), "w", encoding="utf-8") as f:
        json.dump(transcript, f)
    logger.info(f"Transcript saved to {os.path.join(save_dir, f'transcript_{current_time}_{sample.id}.json')}")
'''


@pytest.mark.parametrize("patched, n", [(False, 1), (True, 2)])
def test_sim_vail_counts_two_epochs_of_a_sample_finishing_in_the_same_second(tmp_path, patched, n):
    cfg = load_tasks()["sim_vail"]
    process_results = cfg.process_results.resolve(cfg.task_dir)
    hook = tmp_path / "petri/src/petri/hooks/transcript_hook.py"
    hook.parent.mkdir(parents=True)
    hook.write_text(PETRI_HOOK)
    if patched:  # the task's install step, run where the workspace runs it (the Petri checkout)
        env = {**os.environ, "PATH": f"{os.path.dirname(sys.executable)}{os.pathsep}{os.environ['PATH']}"}
        for cmd in cfg.env["install"]:
            if "transcript_hook.py" in cmd:
                subprocess.run(["bash", "-euo", "pipefail", "-c", cmd], cwd=tmp_path / "petri", env=env, check=True)
    ns = {"os": os, "json": json, "logger": logging.getLogger(__name__)}
    exec(hook.read_text(), ns)

    save_dir = tmp_path / "out/transcripts"
    save_dir.mkdir(parents=True)
    scores = dict.fromkeys([*process_results.__globals__["MH_HARM"], "mh_therapeutic_quality"], 1)
    for epoch in (1, 2):
        sample = SimpleNamespace(id=1, epoch=epoch, uuid=f"uuid{epoch}")
        transcript = {"metadata": {"judge_output": {"scores": scores}, "seed_instruction": "instruction"}}
        ns["save"](sample, str(save_dir), "2026-11-04_12-00-00", transcript)

    res = process_results(tmp_path / "out", {"src": {"sim-vail": str(tmp_path / "sim-vail")}})
    assert len(list(save_dir.iterdir())) == n
    assert res["n"] == n
