import json
from pathlib import Path


def _dim(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "counts"} | d.get("counts", {})


def process_results(output_dir: Path, ctx: dict) -> dict:
    """Pooled VERA scores (scores/scores.json [+ scores_by_risk.json]) -> shared schema."""
    (scores_dir,) = output_dir.glob("pooled/j_*/scores")
    s = json.loads((scores_dir / "scores.json").read_text())
    agg = s["aggregates"]
    metrics = {k: v for k, v in agg.items() if isinstance(v, (int, float))} | {
        k: v for k, v in s["summary"].items() if k.startswith("pct_")}
    breakdown = {"dimension": {k: _dim(v) for k, v in s["dimensions"].items()}}
    if (risk := scores_dir / "scores_by_risk.json").exists():
        levels = json.loads(risk.read_text())["risk_level_scores"]
        breakdown["risk_level"] = {lvl: {"n": v["total_conversations"]} | {
            dim: d["vera_score"] for dim, d in v["dimensions"].items()} for lvl, v in levels.items()}
    return {"metrics": metrics, "breakdown": breakdown, "n": s["summary"]["total_conversations"]}
