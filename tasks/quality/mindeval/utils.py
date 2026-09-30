import json
from pathlib import Path


def process_results(output_dir: Path, ctx: dict) -> dict:
    """judgments.jsonl -> means of `parsed_judgment` (the aggregation in run_benchmark.sh), 1-6 scale."""
    lines = (output_dir / "judgments.jsonl").read_text().splitlines()
    rows = [json.loads(line)["parsed_judgment"] for line in lines]
    means = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    slug = {k: k.lower().replace(" & ", "_and_").replace("-", "_").replace(" ", "_") for k in means}
    return {
        "metrics": {"overall_score": means["Overall score"], "average_score": means["Average score"]},
        "breakdown": {"dimension": {slug[k]: {"mean": v} for k, v in means.items()
                                    if k not in ("Overall score", "Average score")}},
        "n": len(rows),
    }
