"""Set up one EQ-Bench 4 run: its own copy of the reference pool, and the config.yaml the benchmark reads.

    python prepare.py <output_dir> [--holdout MODEL] [--gemini_base_url URL]

The rating is fitted against the repo's bundled pool (results/*.json: every reference model's transcripts and the
pairwise comparisons made when it was added). The benchmark writes the new model's comparisons into its own run
file and rewrites elo_results.json, so each run gets <output_dir>/results: links to the pool files and a private
copy of elo_results.json. A model's rating then depends only on the pool, not on what else was run before it.

Models are identified by their API name, so a model already in the pool cannot be run as a new entrant. `--holdout`
re-runs one from scratch: its own run file and every comparison against it are left out of this run's pool.
"""

import argparse
import json
import os
from pathlib import Path

import yaml

POOL = Path("results")
ELO = "elo_results.json"
GEMINI = "google/gemini-3.1-pro-preview"  # the persona, and one of the pairwise judges
PAIRWISE = {  # the other pairwise judges, as the leaderboard command reaches them (README, config.yaml.example)
    "openai/gpt-5.5": {"api_key": "${OPENROUTER_API_KEY}", "base_url": "https://openrouter.ai/api/v1/chat/completions"},
    "claude-opus-4-6": {"api_key": "${ANTHROPIC_API_KEY}", "base_url": "https://api.anthropic.com"},
}


def evaluated_model(path: Path) -> str | None:
    for run_id, run in json.loads(path.read_text()).items():
        if not run_id.startswith("__") and isinstance(run, dict):
            return (run.get("__meta__") or {}).get("evaluated_model")
    return None


def link_pool(out: Path, target: str, holdout: str | None) -> None:
    pool = out / "results"
    pool.mkdir(parents=True, exist_ok=True)
    pool_models = set()
    for src in sorted(POOL.glob("r*.json")):
        dest = pool / src.name
        model = evaluated_model(src)
        if model == holdout:
            continue
        pool_models.add(model)
        if dest.exists() or dest.is_symlink():
            continue  # a resumed run keeps its pool
        if holdout is None:
            dest.symlink_to(src.resolve())
            continue
        data = json.loads(src.read_text())
        for run_id, run in data.items():
            if not run_id.startswith("__") and isinstance(run, dict) and "__pairwise__" in run:
                run["__pairwise__"] = [c for c in run["__pairwise__"] if holdout not in (c.get("pair") or {}).values()]
        dest.write_text(json.dumps(data))
    if holdout and holdout not in {evaluated_model(p) for p in POOL.glob("r*.json")}:
        raise SystemExit(f"--holdout {holdout} is not a model in the pool")
    if target in pool_models:
        raise SystemExit(f"{target} is already in the EQ-Bench 4 pool; rerun it with --task_args holdout={target}")
    if not (pool / ELO).exists():
        elo = json.loads((POOL / ELO).read_text())
        elo["model_ratings"].pop(holdout, None)
        (pool / ELO).write_text(json.dumps(elo, indent=2))


def write_config(out: Path, gemini_base_url: str) -> None:
    def creds(prefix: str) -> dict:
        base = os.environ[f"{prefix}_BASE_URL"]
        return {"api_key": f"${{{prefix}_API_KEY}}",  # expanded by the benchmark, so no key is written to disk
                "base_url": base if "anthropic.com" in base else f"{base}/chat/completions"}

    target, judge, user = (os.environ[f"{p}_MODEL"] for p in ("TARGET", "JUDGE", "USER"))
    if user != GEMINI:
        raise SystemExit(f"the persona must be {GEMINI}, which played every pool model's users; got {user}")
    model_creds = {**PAIRWISE, judge: creds("JUDGE"), user: {**creds("USER"), "base_url": gemini_base_url},
                   target: creds("TARGET")}
    example = yaml.safe_load(Path("config.yaml.example").read_text())
    config = {**example, "model_creds": model_creds, "enable_model_summary": False}
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--gemini_base_url", default=None, help="Gemini's chat/completions URL (default: the user's)")
    args = parser.parse_args()
    link_pool(args.output_dir, os.environ["TARGET_MODEL"], args.holdout or None)
    write_config(args.output_dir, args.gemini_base_url or f"{os.environ['USER_BASE_URL']}/chat/completions")


if __name__ == "__main__":
    main()
