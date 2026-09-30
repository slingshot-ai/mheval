"""HealthBench-Psych: drives the original repo's own components for a configurable target and judge.

The repo's model list is a hard-coded REGISTRY, so instead of editing it we reuse its plug-in seam:
- `lib.samplers.OpenAICompatSampler` for model access, with REGISTRY's per-model settings where listed;
- `run_eval.load_subset` and `generate.load_examples` for the frozen subset;
- `generate._gen_one` for generation with the repo's bounded retry on empty answers;
- `lib.hb_grade.grade_response`, the HealthBench grader copied verbatim from simple-evals;
- `aggregate.clipped_mean` for the headline score.
Per-tag scores follow simple-evals' HealthBenchEval (example tags = theme, rubric tags = axis).
"""
import argparse
import json
import os
import time
from collections import defaultdict

from mheval.native import import_original, pmap, write_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--subset", default="healthbench-psych-v2")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    run_eval = import_original(f"{a.repo}/eval", "run_eval")
    generate = import_original(f"{a.repo}/eval", "generate", stub=("backends",))
    aggregate = import_original(f"{a.repo}/eval", "aggregate")
    from lib import hb_grade
    from lib.samplers import REGISTRY, OpenAICompatSampler

    _, prompt_ids = run_eval.load_subset(a.subset)
    examples = generate.load_examples(prompt_ids, a.source)
    rows = [examples[p] for p in sorted(examples)][: a.limit]

    class Sampler(OpenAICompatSampler):
        """The repo's sampler, made robust to two endpoint behaviors: rejected temperature/max_tokens for
        reasoning models outside REGISTRY (switch to REGISTRY's settings for such models: temperature=None,
        token_param="max_completion_tokens"), and 200 responses without `choices` (retried with backoff)."""

        def _raw(self, x, tries=3):
            for attempt in range(6):
                try:
                    response = super()._raw(x)
                except RuntimeError as e:  # idempotent: the sampler is shared across worker threads
                    if not tries or "HTTP 400" not in str(e):
                        raise
                    if "temperature" in str(e):
                        self._temp = None
                    elif "max_tokens" in str(e):
                        self._token_param = "max_completion_tokens"
                    else:
                        raise
                    return self._raw(x, tries - 1)
                if response.get("choices"):
                    return response
                time.sleep(min(2 ** attempt, 30))
            raise RuntimeError(f"no choices in response: {str(response)[:200]}")

    def sampler(role):
        """Our endpoint, with the repo's own per-model settings when the model is in its REGISTRY; else the
        repo defaults (temperature 0, max_tokens)."""
        model, params = os.environ[f"{role}_MODEL"], json.loads(os.environ[f"{role}_PARAMS"])
        ref = REGISTRY.get(model.split("/")[-1])
        ref = ref if isinstance(ref, OpenAICompatSampler) else OpenAICompatSampler(model, "", "")
        temperature = params.pop("temperature", ref._temp)
        return Sampler(model, os.environ[f"{role}_BASE_URL"], f"{role}_API_KEY", extra={**ref._extra, **params},
                       temperature=temperature, gen_max=ref._gen_max, token_param=ref._token_param)

    target, judge = sampler("TARGET"), sampler("JUDGE")

    def run(row):
        text, finish = generate._gen_one(target, row["prompt"])  # the repo's bounded retry on empty answers
        items = [hb_grade.RubricItem.from_dict(r) for r in row["rubrics"]]
        score, grades = hb_grade.grade_response(row["prompt"], text, items, judge)
        return {"prompt_id": row["prompt_id"], "response_text": text, "finish_reason": finish, "score": score,
                "example_tags": row["example_tags"], "rubrics": row["rubrics"], "grades": grades}

    samples = pmap(run, rows, a.workers, "healthbench-psych")

    by_tag = defaultdict(list)
    for s in samples:
        for tag in s["example_tags"]:
            by_tag[tag].append(s["score"])
        items = defaultdict(list)
        for r, g in zip(s["rubrics"], s["grades"]):
            for tag in r["tags"]:
                items[tag].append((hb_grade.RubricItem.from_dict(r), g))
        for tag, ig in items.items():
            if (sc := hb_grade.calculate_score([i for i, _ in ig], [g for _, g in ig])) is not None:
                by_tag[tag].append(sc)
    breakdown = defaultdict(dict)
    for tag, scores in sorted(by_tag.items()):
        kind, _, name = tag.partition(":")
        if kind in ("theme", "axis"):
            breakdown[kind][name] = {"clipped_mean_score": aggregate.clipped_mean(scores), "n": len(scores)}
    write_results({"clipped_mean_score": aggregate.clipped_mean([s["score"] for s in samples])},
                  dict(breakdown), len(samples), samples)


if __name__ == "__main__":
    main()
