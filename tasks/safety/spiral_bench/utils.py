import json
from pathlib import Path


def _slug(k: str) -> str:
    return k.replace("-", "_")


def process_results(output_dir: Path, ctx: dict) -> dict:
    """__meta__.scoring_summary -> shared schema. Headline = the judge-averaged row
    (labelled 'overall' when there is a single judge); other rows become the per-judge breakdown."""
    run = json.loads((output_dir / "results_spiral.json").read_text())["mheval"]
    meta = run["__meta__"]
    rows = meta["scoring_summary"]
    head = next((r for r in rows if r["judge"].endswith("(averaged)")), None) or next(
        r for r in rows if r["judge"] == "overall")
    skip = {"model_name", "judge", "score_norm", "ci_low_norm", "ci_high_norm"}
    metrics = {_slug(k): v for k, v in head.items() if k not in skip}
    metrics["off_rails"] = meta.get("final_judgement_summary", {}).get("off-rails", metrics.get("off_rails"))
    n = sum(len(convos) for key, prompts in run.items() if key != "__meta__" for convos in prompts.values())
    return {"metrics": metrics,
            "breakdown": {"judge": {r["judge"]: {_slug(k): v for k, v in r.items() if k not in skip}
                                    for r in rows if r is not head}},
            "n": n}
