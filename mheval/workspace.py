"""Per-benchmark workspace: pinned checkouts of the original repos/data plus an isolated uv venv."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .config import TaskConfig

HOME = Path(os.environ.get("MHEVAL_HOME", Path.home() / ".cache" / "mheval"))


def _run(cmd: str, cwd: Path, env: dict | None = None) -> None:
    print(f"[mheval] $ {cmd}", flush=True)
    subprocess.run(["bash", "-euo", "pipefail", "-c", cmd], cwd=cwd, env=env, check=True)


def source_name(src: dict) -> str:
    return src.get("name") or Path(src.get("repo") or src["url"]).stem


def _fetch_repo(src: dict, dest: Path) -> None:
    dest.mkdir(parents=True)
    filt = "--filter=blob:none " if src.get("sparse") else ""
    cmds = [f"git init -q && git remote add origin {src['repo']}"]
    if src.get("sparse"):
        cmds.append("git sparse-checkout set --no-cone " + " ".join(f"'{p}'" for p in src["sparse"]))
    cmds.append(f"git fetch -q --depth 1 {filt}origin {src['commit']} && git checkout -q FETCH_HEAD")
    _run(" && ".join(cmds), dest)


def _fetch_url(src: dict, dest: Path) -> None:
    dest.mkdir(parents=True)
    target = dest / Path(src["url"]).name
    print(f"[mheval] downloading {src['url']}", flush=True)
    urllib.request.urlretrieve(src["url"], target)
    if src.get("sha256") and hashlib.sha256(target.read_bytes()).hexdigest() != src["sha256"]:
        raise RuntimeError(f"sha256 mismatch for {src['url']}")
    if src.get("extract"):
        with zipfile.ZipFile(target) as z:
            z.extractall(dest)


def venv_env(ws: Path) -> dict[str, str]:
    venv = ws / ".venv"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX")}
    env.update(VIRTUAL_ENV=str(venv), UV_PROJECT_ENVIRONMENT=str(venv),
               PATH=f"{venv / 'bin'}:{os.environ['PATH']}")
    return env


def prepare(cfg: TaskConfig) -> Path:
    """Return the workspace dir, building it on first use. Tasks with identical source+env share one."""
    spec = json.dumps({"source": cfg.source, "env": cfg.env}, sort_keys=True)
    key = hashlib.sha256(spec.encode()).hexdigest()[:10]
    ws = HOME / f"{source_name(cfg.source[0])}-{key}"
    if (ws / ".ready").exists():
        return ws
    if ws.exists():
        shutil.rmtree(ws)  # half-built from an interrupted run
    for src in cfg.source:
        (_fetch_repo if "repo" in src else _fetch_url)(src, ws / source_name(src))
    _run(f"uv venv -q --python {cfg.env['python']} {ws / '.venv'}", ws)
    for cmd in cfg.env.get("install", []):
        _run(cmd, ws / source_name(cfg.source[0]), venv_env(ws))
    (ws / ".ready").write_text(spec)
    return ws
