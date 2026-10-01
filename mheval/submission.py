"""Leaderboard submission files, written next to the results of every full run.

    <output_path>/<model>/submission/
        models/<model>.yaml            # describe the model once (fill in the TODOs)
        results/<model>/<task>.json    # one file per benchmark

Copy the contents of `submission/` into the root of the leaderboard repository and open a pull request.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MODEL_TEMPLATE = """\
# Model card for the leaderboard. Replace every TODO.
#   access: open-weights | public-api | private
#   url: model card, API docs or paper
#   submitted_by: your GitHub handle, e.g. "@octocat"
name: {model}
organization: TODO
access: TODO
url: TODO
submitted_by: TODO
"""


def model_id(model: str) -> str:
    """File-system-safe model id, also used for the results directory (e.g. `anthropic__claude-sonnet-4.5`)."""
    return re.sub(r"[^\w.-]+", "__", model)


def write(run_dir: Path, model: str, task: str, result: dict) -> Path:
    """Write `result` (a task's results.json, including `meta`) into the submission folder."""
    mid = model_id(model)
    root = run_dir / "submission"
    out = root / "results" / mid / f"{task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    card = root / "models" / f"{mid}.yaml"
    if not card.exists():
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(MODEL_TEMPLATE.format(model=model))
    return out
