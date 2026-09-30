"""Run one task: build its workspace, render and execute its command, and collect results.

Every task produces results in the same schema:
    {"metrics":   {name: number},                       # headline numbers; metric_list[0] is primary
     "breakdown": {axis: {level: {name: number}}},      # e.g. dimension / criterion / acuity / risk level
     "n":         int}                                  # evaluated items (conversations, prompts, ...)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jinja2
import jinja2.meta

from .config import ROLES, Role, TaskConfig
from .workspace import prepare, source_name, venv_env

REPO_ROOT = Path(__file__).resolve().parent.parent
_jinja = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
_jinja.filters["merge"] = lambda a, b: {**a, **b}


def render(template: str, ctx: dict) -> str:
    return _jinja.from_string(template).render(**ctx)


def run_task(cfg: TaskConfig, roles: dict[str, Role | None], output_dir: Path, *, limit: int | None,
             gen_kwargs: dict, task_args: dict) -> dict:
    ws = prepare(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    roles = {r: Role.parse(roles.get(r), cfg.roles.get(r)) for r in ROLES}
    gen = {**cfg.generation_kwargs, **gen_kwargs}
    if roles["target"]:  # sampling params for the model under test are request params of the target role
        roles["target"].params = {**gen, **roles["target"].params}
    ctx = {
        **{r: (vars(v) if v else None) for r, v in roles.items()},
        "gen": gen,
        "args": {**cfg.task_args, **task_args},
        "limit": limit,
        "output_dir": str(output_dir.resolve()),
        "task_dir": str(cfg.task_dir.resolve()),
        "src": {source_name(s): str(ws / source_name(s)) for s in cfg.source},
    }
    env = venv_env(ws)
    for r, role in roles.items():
        if role:
            env.update(role.env(r.upper()))
    env.update(MHEVAL_OUTPUT_DIR=ctx["output_dir"],
               PYTHONPATH=str(REPO_ROOT))  # only `mheval.native`; nothing leaks in from the caller
    env.update({k: render(str(v), {**ctx, "env": env}) for k, v in cfg.env_vars.items()})

    if limit and "limit" not in jinja2.meta.find_undeclared_variables(_jinja.parse(cfg.command)):
        print(f"[mheval] WARNING: {cfg.task} does not support --limit; running the full benchmark", flush=True)
    script = render(cfg.command, ctx)
    (output_dir / "command.sh").write_text(script)
    print(f"[mheval] running {cfg.task} -> {output_dir}", flush=True)
    with open(output_dir / "log.txt", "w") as log:
        proc = subprocess.Popen(["bash", "-euo", "pipefail", "-c", script], cwd=ws / source_name(cfg.source[0]),
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
    if proc.wait():
        raise RuntimeError(f"{cfg.task} failed (exit {proc.returncode}); see {output_dir / 'log.txt'}")

    if cfg.process_results:
        res = cfg.process_results.resolve(cfg.task_dir)(output_dir, ctx)
        (output_dir / "results.json").write_text(json.dumps(res, indent=2))
    else:
        res = json.loads((output_dir / "results.json").read_text())
    primary = cfg.metric_list[0]["metric"]
    return {"primary": {"metric": primary, "value": res["metrics"].get(primary),
                        "higher_is_better": cfg.metric_list[0].get("higher_is_better", True)},
            "metrics": res["metrics"], "breakdown": res.get("breakdown", {}), "n": res.get("n"),
            "roles": {r: v["model"] for r in ROLES if (v := ctx[r])}, "output_dir": ctx["output_dir"]}
