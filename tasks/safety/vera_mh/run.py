"""Run VERA-MH's own `vera.py pipeline` with mheval roles routed to OpenAI-compatible endpoints.

VERA picks a provider from substrings of the model name ("gpt", "claude", ...) and takes base URLs from
process-global env vars, so an arbitrary model id on an arbitrary endpoint cannot be named directly.
We register aliases (mheval-target / -user / -judge) in `LLMFactory.create_llm` that build VERA's own
`OpenAILLM` against the role's endpoint (retrying empty replies); everything else (personas, rubric, judge,
scoring) is VERA's.
Roles without an mheval override keep the published config's models (configs/recommended-SI.json).
"""
import json
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from llm_clients import llm_factory  # noqa: E402
from llm_clients.openai_llm import OpenAILLM  # noqa: E402

ALIASES = {f"mheval-{r}": r.upper() for r in ("target", "user", "judge")}
EMPTY_RETRIES = 2  # same bounded retry on empty answers as mheval's own client
_create_llm = llm_factory.LLMFactory.create_llm


class RetryingOpenAILLM(OpenAILLM):
    """VERA's OpenAILLM, retrying empty replies. Some gateways return an empty message on upstream timeouts;
    VERA would pass it on as an empty turn, which the next model's API rejects, and drop the conversation."""

    async def generate_response(self, conversation_history=None):
        text = ""
        for _ in range(EMPTY_RETRIES + 1):
            text = await super().generate_response(conversation_history)
            if text and text.strip():
                break
        return text


def create_llm(model_name, name, role, system_prompt=None, **kwargs):
    if model_name in ALIASES:
        p = ALIASES[model_name]
        kwargs = {k: v for k, v in kwargs.items() if k not in ("model", "name", "prompt_name", "system_prompt")}
        return RetryingOpenAILLM(name, role, system_prompt, os.environ[f"{p}_MODEL"],
                                 base_url=os.environ[f"{p}_BASE_URL"], openai_api_key=os.environ[f"{p}_API_KEY"],
                                 **kwargs)
    return _create_llm(model_name, name, role, system_prompt, **kwargs)


def build_config(base, output, turns, max_concurrent):
    cfg = json.loads(Path(base).read_text())
    gen, judging = cfg["generation"], cfg["judging"]
    gen["chatbot"] = {"name": "mheval-target", "repeats": 1, **json.loads(os.environ["TARGET_PARAMS"])}
    if "USER_MODEL" in os.environ:
        gen["user"] = [{"name": "mheval-user", "repeats": 1, **json.loads(os.environ["USER_PARAMS"])}]
    if "JUDGE_MODEL" in os.environ:
        judging["models"] = [{"name": "mheval-judge", "repeats": 1, **json.loads(os.environ["JUDGE_PARAMS"])}]
    gen.update(output=output, turns=turns, max_concurrent=max_concurrent)
    judging["max_concurrent"] = max_concurrent
    return cfg


if __name__ == "__main__":
    base, output, turns, max_concurrent, *rest = sys.argv[1:]
    os.environ["VERA_RUN_CONFIG"] = json.dumps(build_config(base, output, int(turns), int(max_concurrent)))
    llm_factory.LLMFactory.create_llm = staticmethod(create_llm)
    sys.argv = ["vera.py", "pipeline", *rest]
    runpy.run_path("vera.py", run_name="__main__")
