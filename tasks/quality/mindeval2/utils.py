import json
import statistics
from pathlib import Path


def process_results(output_dir: Path, ctx: dict) -> dict:
    """judge/mindeval2/summary.json -> MHQR (the paper's headline, `mqm.mean`: mean weighted severity per session,
    then over every member x session) and its incident rates; breakdowns by reply severity, issue, and session
    number (from verdicts.jsonl, with the same weights)."""
    judge_dir = output_dir / "judge" / "mindeval2"
    summary = json.loads((judge_dir / "summary.json").read_text())
    coverage, mqm = summary["coverage"], summary["mqm"]
    if coverage["sessions_judged"] != coverage["sessions_expected"] or summary.get("replies", {}).get("pending"):
        raise RuntimeError(f"incomplete run: {coverage}; rerun the same command to finish it")
    weights = mqm["weights"]
    per_session: dict[tuple[str, str], list[int]] = {}
    for line in (judge_dir / "verdicts.jsonl").read_text().splitlines():
        v = json.loads(line)
        if "_error" not in v:
            per_session.setdefault((v["member"], v["episode"]), []).append(weights[v["severity"]])
    by_episode: dict[str, list[float]] = {}
    for (_, episode), w in per_session.items():
        by_episode.setdefault(episode, []).append(statistics.mean(w))
    slug = {s: s.lower().replace(" ", "_") for s in summary["severity_rates"]}
    return {
        "metrics": {"mhqr": mqm["mean"], "mhqr_p99": mqm["p99"],
                    "critical_session_rate": mqm["incidents"]["critical"],
                    "major_or_worse_session_rate": mqm["incidents"]["major_or_worse"]},
        "breakdown": {
            "severity": {slug[s]: {"rate": r} for s, r in summary["severity_rates"].items()},
            "issue": {k: {"rate": r} for k, r in summary["issue_rates"].items()},
            "session": {e: {"mhqr": round(statistics.mean(s), 3), "n": len(s)} for e, s in sorted(by_episode.items())},
        },
        "n": mqm["sessions"],
        "n_replies": summary["n_replies"],
        "judge_errors": summary["errors"],
    }
