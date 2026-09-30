"""Task configuration (one YAML file per task) and model roles."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"
ROLES = ("target", "judge", "user")
OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclasses.dataclass
class Role:
    """An OpenAI-compatible chat endpoint: model id, API root (…/v1), the env var holding its key,
    and extra request params (e.g. reasoning_effort, temperature)."""

    model: str
    base_url: str = OPENAI_BASE_URL
    api_key_env: str = "OPENAI_API_KEY"
    params: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def parse(cls, spec: str | dict | None, default: dict | None = None) -> Role | None:
        """Merge `k=v,k=v` (or a dict) over a task default. Unknown keys become request params."""
        spec = parse_kv(spec) if isinstance(spec, str) else dict(spec or {})
        base = dict(default or {})
        if "model" in spec:  # default params are model-specific
            base = {k: v for k, v in base.items() if k in ("base_url", "api_key_env")}
        if "base_url" in spec:  # never send one provider's key to another endpoint
            base.pop("api_key_env", None)
            spec.setdefault("api_key_env", "")
        merged = {**base, **spec}
        if not merged.get("model"):
            return None
        fields = {"model", "base_url", "api_key_env"}
        return cls(**{k: v for k, v in merged.items() if k in fields},
                   params={k: v for k, v in merged.items() if k not in fields})

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "EMPTY") if self.api_key_env else "EMPTY"

    def env(self, prefix: str) -> dict[str, str]:
        return {f"{prefix}_MODEL": str(self.model), f"{prefix}_BASE_URL": self.base_url.rstrip("/"),
                f"{prefix}_API_KEY": self.api_key, f"{prefix}_PARAMS": json.dumps(self.params)}


class _Function(str):
    """YAML tag `!function module.fn`, resolved against `module.py` next to the task file."""

    def resolve(self, task_dir: Path) -> Callable:
        module, fn = self.rsplit(".", 1)
        spec = importlib.util.spec_from_file_location(f"mheval_task_{module}", task_dir / f"{module}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, fn)


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!function", lambda loader, node: _Function(loader.construct_scalar(node)))


@dataclasses.dataclass
class TaskConfig:
    task: str
    tag: list[str]
    source: list[dict]                 # [{repo, commit, sparse?} | {url, sha256, extract?}], first = working dir
    env: dict                          # {python, install: [shell cmds]}
    roles: dict[str, dict]             # default judge/user endpoints (target comes from --model_args)
    generation_kwargs: dict            # target sampling defaults, overridable via --gen_kwargs
    task_args: dict                    # benchmark knobs, overridable via --task_args
    env_vars: dict[str, str]           # templated env for the command
    command: str                       # templated bash, run in the first source dir inside the task env
    process_results: _Function | None  # output_dir -> results dict; None = command wrote results.json
    metric_list: list[dict]            # [{metric, higher_is_better}], first = primary
    metadata: dict
    task_dir: Path = dataclasses.field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path) -> TaskConfig:
        raw = yaml.load(path.read_text(), Loader=_Loader)
        fields = {f.name for f in dataclasses.fields(cls)} - {"task_dir"}
        if missing := fields - raw.keys():
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        if extra := raw.keys() - fields:
            raise ValueError(f"{path}: unknown keys {sorted(extra)}")
        return cls(**raw, task_dir=path.parent)


def load_tasks(include_paths: list[str] = ()) -> dict[str, TaskConfig]:
    tasks = {}
    for root in [TASKS_DIR, *map(Path, include_paths)]:
        for path in sorted(root.rglob("*.yaml")):
            cfg = TaskConfig.load(path)
            tasks[cfg.task] = cfg
    return tasks


def select(tasks: dict[str, TaskConfig], names: list[str]) -> list[TaskConfig]:
    """Resolve task names and tags (e.g. `safety`) to configs, preserving order."""
    out = []
    for name in names:
        hits = [tasks[name]] if name in tasks else [t for t in tasks.values() if name in t.tag]
        if not hits:
            raise SystemExit(f"Unknown task or tag: {name}. See `mheval --tasks list`.")
        out += [t for t in hits if t not in out]
    return out


def parse_kv(s: str | None) -> dict[str, Any]:
    """`a=1,b=x,c=` -> {'a': 1, 'b': 'x', 'c': ''}. A piece without `=` continues the previous value, so
    `model=m1,m2` keeps a comma list. Values are YAML scalars; model ids and URLs are kept verbatim."""
    raw: dict[str, str] = {}
    key = None
    for piece in (s or "").split(","):
        if "=" in piece:
            key, val = piece.split("=", 1)
            raw[key] = val
        elif key and piece:
            raw[key] += "," + piece
    return {k: v if k in ("model", "base_url", "api_key_env") or v == "" else yaml.safe_load(v)
            for k, v in raw.items()}
