"""CBT-Bench Level II: the original prompt builders and scorers, with generation over an OpenAI-compatible API.

The repo only generates through local vLLM (a fixed HF_MODELS list) or Azure OpenAI, so we import its
`task{2,3,4}_prompts` and `check_generation` and swap in our client. vllm/torch/transformers are imported
at module level but unused on this path, so they are stubbed rather than installed.

Scoring follows the repo's two steps. `check_generation` parses the raw answer strictly (bare letters,
comma-separated). `gpt_check.py` then re-extracts the chosen letters from every answer with an LLM
(gpt-4o at temperature 0.7 in the repo) using its per-task prompts and `utils.extract_choice`. The headline
metrics use the checked answers; the strict parse is reported with a `_strict` suffix.

Level I (qa) is omitted: the released qa_test.json has no answer key. Level III has no code in the repo.
"""
import argparse
import os
from pathlib import Path

from mheval.native import Chat, import_original, pmap, write_results

SUBTASKS = {  # name: (module, prompt fn, label field, labels constant)
    "distortions": ("eval_task2", "task2_prompts", "distortions", "TASK2_LABELS"),
    "core_major": ("eval_task3", "task3_prompts", "core_belief_major", "TASK3_LABELS"),
    "core_fine": ("eval_task4", "task4_prompts", "core_belief_fine_grained", "TASK4_LABELS"),
}


def load_answer_check(repo: str) -> dict:
    """Definitions from gpt_check.py. The script reads a hard-coded result file at module level
    (`file = '...'`), so only the code above that line is executed."""
    source = Path(repo, "gpt_check.py").read_text()
    namespace: dict = {}
    exec(compile(source.split("\nfile = ")[0], "gpt_check.py", "exec"), namespace)
    return namespace


def extract(utils, text: str) -> str:
    """`utils.extract_choice`, treating "Answer: []" (no letters chosen) as no choice: the repo's regex
    leaves both groups empty there and raises on `None.strip()`."""
    try:
        return (utils.extract_choice(text) or "").strip()
    except AttributeError:
        return ""


def scores(mod, answers, texts, labels):
    from sklearn.metrics import f1_score, precision_score, recall_score

    f1, preds, follow = mod.check_generation(texts, answers)
    return preds, {
        "weighted_f1": f1,
        # the paper reports precision/recall "averaged by class portion" (Table 5)
        "weighted_precision": precision_score(answers, preds, average="weighted", zero_division=0),
        "weighted_recall": recall_score(answers, preds, average="weighted", zero_division=0),
        "format_rate": sum(follow) / len(follow),
    }, dict(zip(labels, map(float, f1_score(answers, preds, average=None, zero_division=0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    a.repo = os.path.abspath(a.repo)
    os.chdir(f"{a.repo}/CBT_LLM_data")  # constants.CBT_DATA_DIR = '' -> data paths are cwd-relative
    constants = import_original(a.repo, "constants")
    utils = import_original(a.repo, "utils", stub=("vllm", "torch", "transformers"))
    check = load_answer_check(a.repo)

    target, extractor = Chat("target"), Chat("judge")
    metrics, breakdown, samples, n = {}, {"label_f1": {}, "label_f1_strict": {}}, [], 0
    for name, (module, prompt_fn, field, labels_const) in SUBTASKS.items():
        mod = import_original(a.repo, module, stub=("vllm", "torch", "transformers"))
        labels = getattr(constants, labels_const)
        data = mod.load_json(getattr(constants, f"TASK{module[-1]}_TEST"))[: a.limit]
        answers = [[int(lab in item[field]) for lab in labels] for item in data]

        gens = pmap(target, getattr(mod, prompt_fn)(data, a.shot, return_conversation=True), a.workers, name)
        _, strict, per_label_strict = scores(mod, answers, gens, labels)

        check_convs = check["task234_check_prompts"]([{"generation": g} for g in gens], module[-1])
        extracted = pmap(extractor, check_convs, a.workers, f"{name} answer check")
        choices = [extract(utils, e) for e in extracted]
        preds, checked, per_label = scores(mod, answers, choices, labels)

        metrics |= {f"{name}_{k}": v for k, v in checked.items()}
        metrics |= {f"{name}_{k}_strict": v for k, v in strict.items()}
        breakdown["label_f1"][name], breakdown["label_f1_strict"][name] = per_label, per_label_strict
        samples += [{"subtask": name, "id": d.get("id"), "generation": g, "checked_choice": c, "prediction": p,
                     "answer": y} for d, g, c, p, y in zip(data, gens, choices, preds, answers)]
        n += len(data)
    for suffix in ("", "_strict"):
        metrics[f"mean_weighted_f1{suffix}"] = sum(metrics[f"{s}_weighted_f1{suffix}"] for s in SUBTASKS) / 3
    write_results(metrics, breakdown, n, samples)


if __name__ == "__main__":
    main()
