# Mental Health Evaluation Harness

**Mental-health LLM benchmarks, run from their original repositories behind one interface.**

[![CI](https://github.com/slingshot-ai/mheval/actions/workflows/ci.yml/badge.svg)](https://github.com/slingshot-ai/mheval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Benchmarks](https://img.shields.io/badge/benchmarks-9-informational.svg)](#benchmarks)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

The Mental Health Evaluation Harness (`mheval`) evaluates any language model on nine published therapy and mental-health benchmarks (quality and safety) with a single command. Each benchmark runs from **its original repository**: pinned to a commit, installed in its own isolated `uv` environment, and invoked through its own prompts, judges and scoring code. A thin adapter is added only where the original cannot run an arbitrary model.

- **Faithful.** Original code, pinned commits, sha256-verified data, and each paper's evaluation settings.
- **Any endpoint.** The model under test, judges and user simulators are each any OpenAI-compatible API: OpenAI, OpenRouter, vLLM, SGLang, LiteLLM, and so on.
- **One result schema.** Every task reports a headline metric, all metrics, and per-dimension breakdowns in the same JSON shape.
- **Declarative tasks.** One YAML file per benchmark, all with the same keys; adding a benchmark rarely needs new code.

## Installation

Requires Python ≥ 3.10, [`uv`](https://docs.astral.sh/uv/) and `git`.

```bash
git clone https://github.com/slingshot-ai/mheval.git
cd mheval
uv venv && uv pip install -e .
```

On first use, each task clones its source repository, downloads its data and builds its environment under `$MHEVAL_HOME` (default `~/.cache/mheval`). API keys are read from environment variables (see [Usage](#usage)).

## Quick start

```bash
mheval --tasks list                                  # available tasks and tags

# a local vLLM server, all quality benchmarks
mheval --tasks quality --model_args model=Qwen/Qwen3-8B,base_url=http://localhost:8000/v1

# an API model, safety benchmarks, first 2 items each (smoke test)
mheval --tasks safety --model_args model=gpt-5.2 --limit 2

# override the judge, user simulator, sampling or benchmark settings
mheval --tasks mindeval \
  --model_args model=openai/gpt-5.2,base_url=https://openrouter.ai/api/v1,api_key_env=OPENROUTER_API_KEY \
  --judge_args model=gpt-5.4,reasoning_effort=low --gen_kwargs temperature=0.7 --task_args n_turns=10

python -m mheval.report results/                     # Markdown summary of all runs
```

## Benchmarks

### Therapy and mental health: quality

| Benchmark | Task | What it tests | Default judge / user simulator | Headline metric | Breakdowns |
|---|---|---|---|---|---|
| [MentalHealthBench](https://cdn.openai.com/ctf-cdn/MentalHealthBench_A_Comprehensive_Benchmark_of_AI_Capabilities_in_Realistic_Mental_Health_Conversations.pdf) (OpenAI) | `mentalhealthbench` | Next response in 1,215 synthetic conversations across acuity levels and user types | `gpt-5.6-sol`, high reasoning | `task_clipped_score` (0–1) | 10 behavior axes, acuity, user profile, language, prior context |
| [MindEval](https://github.com/SWORDHealth/mind-eval) (Sword Health) | `mindeval` | Full therapy sessions (20 turns) with a simulated patient | judge `claude-sonnet-4.5`, patient `claude-haiku-4.5` | `average_score` (1–6) | 5 clinical criteria |
| [HealthBench-Psych](https://github.com/mindbench-ai/healthbench-psych) | `healthbench_psych` | Clinician-adjudicated mental-health subset of HealthBench | `gpt-4.1-2025-04-14` | `clipped_mean_score` (0–1) | theme, rubric axis |
| [EQ-Bench 3](https://github.com/EQ-bench/eqbench3) | `eqbench3` | Emotional intelligence in multi-turn role-play and transcript analysis | `claude-opus-4.6` | `rubric_score` (0–100), `elo_normalized` | scenario type, 18 rubric criteria |
| [CounselBench](https://github.com/llm-eval-mental-health/CounselBench) | `counselbench_eval` | Single-turn counseling answers (EVAL) | `gpt-4.1` | `overall` (1–5) | 6 metrics, topic |
| | `counselbench_adv` | 120 adversarial questions (Adv) | `gpt-4.1` | `failure_rate` (0–1) | 6 failure modes |
| [CBT-Bench](https://github.com/mianzhang/CBT-Bench) | `cbt_bench` | Cognitive distortion and core-belief classification (Level II) | answer extraction `gpt-4o` | `mean_weighted_f1` (0–1) | subtask precision / recall / F1, per-label F1 |

### Therapy and mental health: safety

| Benchmark | Task | What it tests | Default judge / user simulator | Headline metric | Breakdowns |
|---|---|---|---|---|---|
| [VERA-MH](https://github.com/SpringCare/VERA-MH) (Spring Health) | `vera_mh` | Suicide-risk detection and response, 100 clinician-written personas, up to 30 turns | users `gpt-5.2` + `claude-opus-4-5`, judge `gpt-5.4` | `vera_score` (0–100) | 5 rubric dimensions, persona risk level |
| [Spiral-Bench](https://github.com/sam-paech/spiral-bench) | `spiral_bench` | Sycophancy, delusion reinforcement, harmful advice and unwarranted referrals over 20-turn chats | user `kimi-k2.5`, judge `gpt-5-2025-08-07` | `score_0_100` | 17 behaviors + off-rails, per judge |
| [SIM-VAIL](https://github.com/veithweilnhammer/sim-vail) | `sim_vail` | Simulated vulnerable users (5 phenotypes × 6 intentions) in multi-turn audits | auditor `claude-sonnet-4.5`, judge `claude-opus-4.5` | `mh_harm_mean` (1–10) | 39 judge dimensions, phenotype, intention |

## Results

Full benchmarks with each task's default judges and user simulators, run in September 2026. Arrows show the better direction; see [Benchmarks](#benchmarks) for scales.

#### Quality

| Model | cbt_bench<br>`mean_weighted_f1` ↑ | counselbench_adv<br>`failure_rate` ↓ | counselbench_eval<br>`overall` ↑ | eqbench3<br>`rubric_score` ↑ | healthbench_psych<br>`clipped_mean_score` ↑ | mentalhealthbench<br>`task_clipped_score` ↑ | mindeval<br>`average_score` ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `anthropic/claude-sonnet-4.5` | 0.600 | 0.567 | **5.000** | 72.35 | 0.433 | 0.402 | 3.718 |
| `google/gemini-2.5-pro` | **0.630** | 0.536 | **5.000** | 74.25 | 0.484 | 0.295 | **3.756** |
| `gpt-4o-2024-08-06` | 0.602 | **0.367** | 4.960 | 39.80 | 0.276 | 0.326 | 3.169 |
| `gpt-5-2025-08-07` | 0.617 | 0.494 | **5.000** | **77.70** | **0.662** | **0.444** | 3.512 |
| `qwen/qwen3.5-27b` | 0.592 | 0.489 | 4.990 | 65.10 | 0.541 | 0.367 | 3.598 |

#### Safety

| Model | sim_vail<br>`mh_harm_mean` ↓ | spiral_bench<br>`score_0_100` ↑ | vera_mh<br>`vera_score` ↑ |
| --- | ---: | ---: | ---: |
| `anthropic/claude-sonnet-4.5` | **1.031** | **66.90** | 56.03 |
| `google/gemini-2.5-pro` | 3.599 | 44.60 | 30.73 |
| `gpt-4o-2024-08-06` | 2.911 | 54.60 | 28.18 |
| `gpt-5-2025-08-07` | 2.158 | 60.80 | **62.29** |
| `qwen/qwen3.5-27b` | 1.759 | 42.80 | 39.58 |

Models under test:
- `gpt-4o-2024-08-06` and `gpt-5-2025-08-07` through the OpenAI API, at default reasoning effort.
- `anthropic/claude-sonnet-4.5`, `google/gemini-2.5-pro` and `qwen/qwen3.5-27b` through OpenRouter.

Bold marks the best score per benchmark.

MindEval follows its paper in running reasoning-capable models at high effort: gpt-5, claude-sonnet-4.5, gemini-2.5-pro and qwen3.5-27b use `reasoning_effort=high`. qwen3.5-27b's long reasoning occasionally exceeds a benchmark's output cap and yields no answer, which is scored as given (e.g. 6% of CBT-Bench items); its VERA-MH score covers 199 of 200 conversations. On CounselBench EVAL, the LLM judge rates nearly every answer from these models at the top of its scale, so `overall` separates them little. Per-dimension, per-criterion and per-group breakdowns are in each task's `results.json`.

To reproduce a row:

```bash
mheval --tasks quality,safety --model_args model=gpt-5-2025-08-07
mheval --tasks mindeval --model_args model=gpt-5-2025-08-07 --gen_kwargs reasoning_effort=high
```

## Usage

| Flag | Description |
|---|---|
| `--tasks` | Comma-separated task names or tags (`quality`, `safety`), or `list`. |
| `--model_args` | Model under test: `model=<id>[,base_url=<…/v1>][,api_key_env=<VAR>][,<request param>=<value>]`. Defaults to `https://api.openai.com/v1` with `OPENAI_API_KEY`. |
| `--judge_args`, `--user_args` | Override a task's default judge or user simulator (same syntax). Changing `model` drops model-specific defaults such as `reasoning_effort`. Changing `base_url` drops the default key, so no key is sent to a different provider. A comma-separated `model=a,b,c` stays one value; Spiral-Bench treats it as a judge ensemble. |
| `--gen_kwargs` | Sampling parameters for the model under test, e.g. `temperature=0,max_tokens=2048`, where the benchmark lets the caller set them. |
| `--task_args` | Benchmark settings declared in each task's `task_args`: turns, samples, subset, concurrency, … |
| `--limit N` | Evaluate the first N items (conversations, personas, prompts). EQ-Bench has no subset option and always runs all 45 scenarios. |
| `--output_path` | Default `results/`. |
| `--include_path` | Additional directory of task YAML files. |
| `--setup_only` | Fetch sources and build task environments without running. |

API keys are read from the environment variables named by `api_key_env`. The defaults use `OPENAI_API_KEY` and `OPENROUTER_API_KEY`, plus `ANTHROPIC_API_KEY` for VERA-MH's published user simulator. Keys are passed to benchmark processes through environment variables, never on the command line.

### Outputs

`results/<model>/<task>/` holds everything a task produced: the benchmark's own raw outputs (transcripts, judgments, logs), `command.sh`, `log.txt` and `results.json`. `results/<model>/results_<timestamp>.json` collects all tasks of one invocation. Every task reports the same schema:

```jsonc
{
  "primary":   {"metric": "average_score", "value": 3.7565, "higher_is_better": true},
  "metrics":   {"average_score": 3.7565, "overall_score": 3.7565},
  "breakdown": {"dimension": {"clinical_accuracy_and_competence": {"mean": 3.745}, "...": {}}},
  "n": 50
}
```

EQ-Bench 3 and Spiral-Bench resume from an existing task directory; delete it to start over.

## Task configuration

Each task is one YAML file, and every file has the same keys:

```yaml
task: spiral_bench
tag: [safety]
source:                 # pinned inputs; the first entry is the working directory
  - repo: https://github.com/sam-paech/spiral-bench
    commit: 35411a02d39410cfb8b722a965940784a81cee57
    sparse: [/*.py, /data/, /prompts/, /user_instructions/]
  # - {url: https://…/data.zip, sha256: …, extract: true}
env:                    # isolated environment, built once per distinct (source, env)
  python: "3.12"
  install: [uv pip install requests python-dotenv tqdm numpy pandas]
roles:                  # default judge / user simulator; the model under test comes from --model_args
  judge: {model: gpt-5-2025-08-07, base_url: https://api.openai.com/v1, api_key_env: OPENAI_API_KEY}
generation_kwargs: {}   # sampling defaults for the model under test
task_args: {num_turns: 20, num_prompts: 30, parallelism: 8}
env_vars: {JUDGE_BASE_URL: "{{ judge.base_url }}/chat/completions", …}
command: python main.py --evaluated-model {{ target.model }} …
process_results: !function utils.process_results   # raw outputs -> result schema; null if the command writes results.json
metric_list: [{metric: score_0_100, higher_is_better: true}, …]   # the first entry is the headline
metadata: {version: 1.2, description: …, paper: …, homepage: …, implementation: original}
```

`command` and `env_vars` are Jinja templates with `target`, `judge`, `user` (`model`, `base_url`, `params`), `gen`, `args`, `limit`, `output_dir`, `task_dir`, `src.<source name>` and `env`. `metadata.implementation` is one of:
- `original`: the benchmark's own entry point.
- `original + adapter`: the benchmark's own prompts, judges and scoring, driven by a small adapter so any endpoint can be used.
- `native`: no official code exists.

```
mheval/               CLI, configuration, workspaces, runner, report; native.py is shared by the adapters
tasks/<tag>/<task>/   <task>.yaml, plus utils.py (result parsing) or harness.py (adapter) where needed
tests/
```

To add a benchmark, add a YAML file in the same shape. If the original can run an arbitrary OpenAI-compatible model, call it from `command` and parse its outputs in `process_results`. Otherwise, write a `harness.py` that imports the original code and writes `results.json` with `mheval.native.write_results`.

## Implementation notes

Where a task departs from the benchmark's own scripts, it is listed here.

- **MentalHealthBench.** OpenAI released the data but no evaluation code. The task follows the paper (§2.3, Appendix B.2):
  - 4 samples per conversation, generated with each API's default settings.
  - Every rubric item graded independently by `gpt-5.6-sol` at high reasoning effort, with the published grader prompt.
  - Signed, task-clipped and decomposed scores computed per Eqs. 1–3.

  The paper does not specify how the conversation is shown to the grader; it is rendered as `role: content` blocks. Axis scores average over the conversations that contain that axis.
- **MindEval.** Runs the repo's interaction and judgment scripts with `run_benchmark.sh` settings: 10 exchanges, which is 20 turns in the paper's counting, over 50 patient profiles. It uses the published `v0_1` clinician prompt, because the script's `custom` template is an unfilled placeholder. The patient and judge are reached through OpenRouter rather than Vertex AI. The paper runs reasoning models at high reasoning effort, so the reported results for gpt-5, claude-sonnet-4.5 and gemini-2.5-pro use `--gen_kwargs reasoning_effort=high`. The headline is the paper's `average_score`.
- **HealthBench-Psych.** The repo's models are a fixed registry, so the adapter drives its own components:
  - `OpenAICompatSampler` for model access, reusing the registry's per-model settings where the model is listed.
  - The frozen subset.
  - The HealthBench grader, copied verbatim from simple-evals.
  - The clipped-mean aggregation.

  The repo reports a three-judge panel (`gpt-4.1`, `claude-haiku-4.5`, `gemini-2.5-flash`). This task uses one judge per run, `gpt-4.1` by default; run once per `--judge_args` to reproduce the panel.
- **EQ-Bench 3.** Runs `eqbench3.py` unchanged: rubric scoring plus pairwise ELO against the bundled leaderboard. Sampling is fixed by the benchmark. `--task_args elo=false` skips the ELO pass, which takes the most judge calls.
- **CounselBench.** The repo's model classes accept only a fixed list of model names and read keys from `config.json`, and its data is not in the repository. The adapter:
  - reads the released Hugging Face data at pinned revisions;
  - implements the repo's `BaseModel` interface on an OpenAI-compatible client (same system prompt, temperature, token limit and JSON mode);
  - calls the repo's own length-constrained regeneration, judge prompts, score parsing, averaging and adversarial judge.

  EVAL uses the README's settings: temperature 0.7, and regeneration until the answer is at most 250 words, up to 20 attempts, after which the last answer is kept. The Adv few-shot example for each failure mode comes from the released human-annotation data, as the repo's example file is not published. As in the repo, the Adv judge receives the issue key (e.g. `apathetic`). EVAL judge outputs that cannot be parsed are excluded and counted in `judge_parse_failures`.
- **CBT-Bench.** Level II, run at temperature 0.0 as in the paper, scored in the repository's two steps. First, `check_generation` parses the raw answer strictly as comma-separated letters. Then `gpt_check.py` re-extracts the chosen letters from every answer with an LLM (gpt-4o at temperature 0.7, the repo's default). Headline metrics use the re-extracted answers, and the strict parse is also reported (`*_strict`); the two differ for models that explain their choice instead of answering with letters only. Weighted precision and recall are reported alongside F1, as in the paper. Level I is omitted because the released test set has no answer key, and Level III has no code in the repository. The repo imports `vllm`, `torch` and `transformers` at module level without using them on this path, so they are stubbed rather than installed.
- **VERA-MH.** Runs `vera.py pipeline` with the published `configs/recommended-SI.json` and pools the per-simulator evaluations with `scripts/pool_vera_scores.py`, as the README describes. A small launcher registers the model under test, and any overridden user simulator or judge, in VERA's provider factory, so they can use any OpenAI-compatible endpoint. The launcher also retries empty replies up to twice: VERA would otherwise pass an empty turn to the next model, whose API rejects it, and drop the conversation.
- **Spiral-Bench.** Runs `main.py` with the leaderboard's 20 turns. The README's default user simulator, `kimi-k2`, is not reliably served any more, so its successor `kimi-k2.5` is used. The judge is the README example, `gpt-5-2025-08-07`; the v1.2 leaderboard averaged three judges, each on its own provider endpoint.

  `api_client.py` sets reasoning per model (e.g. minimal for `gpt-5-2025-08-07` and `google/gemini-2.5-pro`), then disables reasoning for every request whose URL starts with `https://openrouter.ai/api`. OpenRouter now rejects disabled reasoning for Gemini 2.5 Pro, so that model is evaluated through `https://www.openrouter.ai/api/v1`, where the benchmark's own per-model setting applies.
- **SIM-VAIL.** The repository provides a Petri override bundle rather than a runner. The task pins Petri to `dc9fba3`, the last commit before the published runs, applies the bundle, and runs `inspect eval petri/audit` with 3 epochs of the 30 instructions. `mh_harm_mean`, the mean of the ten harm-oriented mental-health dimensions, is a summary added here; every judge dimension is reported.
- **API robustness.** Adapters that call the API directly handle a few endpoint behaviors, logging each one. An endpoint that rejects `temperature` gets the API default, and `max_tokens` is sent as `max_completion_tokens`. An answer cut off at a benchmark's output cap after hidden reasoning is retried with 16k extra tokens; the caps (e.g. CounselBench's 1,024) were set for non-reasoning models, where they limit only the visible answer. Empty answers other than refusals, and generations that fail upstream, are retried up to twice, following HealthBench-Psych's generation code. Each request has a 15-minute deadline.

## Contributing

Issues and pull requests are welcome. To set up a development environment and run the checks that CI runs:

```bash
uv pip install -e . pytest ruff
pytest && ruff check .
```

To add a benchmark, follow [Task configuration](#task-configuration). Please keep each benchmark's original code untouched, and put anything harness-specific in the task's own directory.

## Acknowledgements

`mheval` only orchestrates. The benchmarks, data, prompts and scoring belong to their authors: [MentalHealthBench](https://cdn.openai.com/ctf-cdn/MentalHealthBench_A_Comprehensive_Benchmark_of_AI_Capabilities_in_Realistic_Mental_Health_Conversations.pdf), [MindEval](https://github.com/SWORDHealth/mind-eval), [HealthBench-Psych](https://github.com/mindbench-ai/healthbench-psych), [EQ-Bench 3](https://github.com/EQ-bench/eqbench3), [CounselBench](https://github.com/llm-eval-mental-health/CounselBench), [CBT-Bench](https://github.com/mianzhang/CBT-Bench), [VERA-MH](https://github.com/SpringCare/VERA-MH), [Spiral-Bench](https://github.com/sam-paech/spiral-bench), [SIM-VAIL](https://github.com/veithweilnhammer/sim-vail) (with [Petri](https://github.com/safety-research/petri)). Please cite the original papers when reporting results.

## License

MIT. Each benchmark keeps the license of its original repository and dataset.
