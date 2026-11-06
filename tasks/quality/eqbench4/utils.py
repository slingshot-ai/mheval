import json
import statistics
from pathlib import Path


def process_results(output_dir: Path, ctx: dict) -> dict:
    """results/elo_results.json -> the model's rating against the pool (the leaderboard's score, with its bootstrap
    95% interval); results/mheval.json -> its share of the pairwise comparisons won and its mean behavioural-trait
    scores (informational: neither better nor worse when higher)."""
    target, expected = ctx["target"]["model"], ctx["limit"] or 120
    run = json.loads((output_dir / "results" / "mheval.json").read_text())["r1"]
    convos = [c for k, v in run.items() if not k.startswith("__") for c in v]
    done = [c for c in convos if c.get("completed")]
    if len(done) != expected:
        raise RuntimeError(f"incomplete run: {len(done)} of {expected} conversations finished; rerun to resume")
    rating = json.loads((output_dir / "results" / "elo_results.json").read_text())["model_ratings"].get(target)
    if rating is None:
        raise RuntimeError(f"{target} has no rating; the ELO stage failed (see log.txt); rerun to resume")
    won = [c["aggregated"]["fraction_for_model_a"] if c["pair"]["model_a"] == target
           else 1 - c["aggregated"]["fraction_for_model_a"]
           for c in run.get("__pairwise__", []) if "error" not in c]
    traits: dict[str, list[float]] = {}
    for c in done:
        for k, v in (c.get("performance_judge_results") or {}).items():
            if k not in ("num_judges", "per_judge") and isinstance(v, (int, float)):
                traits.setdefault(k, []).append(v)
    return {
        "metrics": {"elo": rating["elo_norm"], "elo_ci_low": rating["ci_low_norm"],
                    "elo_ci_high": rating["ci_high_norm"], "pairwise_win_fraction": statistics.mean(won)},
        "breakdown": {"trait": {k: {"score": round(statistics.mean(v), 3)} for k, v in sorted(traits.items())}},
        "n": len(done),
        "n_comparisons": len(won),
    }
