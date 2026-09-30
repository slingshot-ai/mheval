"""Helpers for the thin native layers. Stdlib only: they run inside each benchmark's own env.

Roles arrive as env vars set by the runner: {TARGET,JUDGE,USER}_{MODEL,BASE_URL,API_KEY,PARAMS}
(PARAMS = JSON request params, incl. --gen_kwargs for the target), plus MHEVAL_OUTPUT_DIR.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
import types
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Chat:
    """Minimal OpenAI-compatible /chat/completions client.

    Retries transient errors with backoff and empty non-refusal answers up to twice, and adapts once per
    client to reasoning models: a rejected
    `temperature`/`top_p` is dropped, `max_tokens` becomes `max_completion_tokens`, and an answer cut off at
    the token cap after hidden reasoning (which counts against the same cap) is retried with extra room.
    """

    REASONING_HEADROOM = 16000
    DEADLINE = 900  # seconds per request, including the response body
    EMPTY_RETRIES = 2  # as HealthBench-Psych's generation (`generate._gen_one`)

    def __init__(self, role: str, **defaults):
        p = role.upper()
        self.model = os.environ[f"{p}_MODEL"]
        self.url = os.environ[f"{p}_BASE_URL"].rstrip("/") + "/chat/completions"
        self.key = os.environ.get(f"{p}_API_KEY", "EMPTY")
        self.defaults = {**defaults, **json.loads(os.environ.get(f"{p}_PARAMS", "{}"))}
        self.rejected: set[str] = set()   # params the endpoint refused; learned once, shared across threads
        self.headroom = 0                 # extra tokens added to any cap after an empty, truncated answer

    def _payload(self, messages: list[dict], kw: dict) -> dict:
        payload = {"model": self.model, "messages": messages, **self.defaults, **kw}
        for param in self.rejected:
            value = payload.pop(param, None)
            if param == "max_tokens" and value is not None:
                payload["max_completion_tokens"] = value
        for cap in ("max_tokens", "max_completion_tokens"):
            if cap in payload:
                payload[cap] += self.headroom
        return payload

    def _log(self, msg: str) -> None:
        print(f"[mheval] {self.model}: {msg}", flush=True)

    def __call__(self, messages: list[dict] | str, retries: int = 8, **kw) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        attempt = empty_retries = 0
        while True:
            headroom = self.headroom
            payload = self._payload(messages, kw)
            req = urllib.request.Request(self.url, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            try:
                response = json.loads(_read(req, self.DEADLINE))
                choice = response["choices"][0]
                if choice.get("finish_reason") == "error":  # the provider failed mid-generation
                    raise ProviderError(json.dumps(choice.get("error") or choice)[:200])
                content = choice["message"].get("content") or ""
                usage = response.get("usage") or {}
                reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                capped = {"max_tokens", "max_completion_tokens"} & payload.keys()
                cut_by_reasoning = choice.get("finish_reason") == "length" and (reasoning or not content)
                if cut_by_reasoning and capped and not headroom:
                    if not self.headroom:
                        self.headroom = self.REASONING_HEADROOM
                        self._log(f"answer cut off at the token cap by hidden reasoning; adding {self.headroom} tokens")
                    continue
                if not content.strip() and choice.get("finish_reason") not in ("content_filter", "length") \
                        and empty_retries < self.EMPTY_RETRIES:
                    empty_retries += 1  # bounded retry on empty, non-refusal answers
                    self._log(f"empty answer (finish_reason={choice.get('finish_reason')}), retry {empty_retries}")
                    continue
                return content
            except urllib.error.HTTPError as e:
                body = e.read()
                param = _rejected_param(body) if e.code == 400 else None
                if param in ("temperature", "top_p", "max_tokens") and param in payload:
                    if param not in self.rejected:
                        self.rejected.add(param)
                        self._log(f"endpoint rejected `{param}`; using the API default")
                    continue
                attempt += 1
                if not (e.code in (408, 409, 429) or e.code >= 500) or attempt == retries:  # any 5xx is transient
                    raise RuntimeError(f"{self.model}: HTTP {e.code} {body[:500]!r}") from e
                self._log(f"HTTP {e.code}, retry {attempt}/{retries}")
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ProviderError) as e:
                attempt += 1  # network errors, timeouts, and 200 responses carrying an error
                if attempt == retries:
                    raise
                self._log(f"{type(e).__name__}: {str(e)[:200]}, retry {attempt}/{retries}")
            time.sleep(min(2 ** attempt, 60))


class ProviderError(Exception):
    """A 200 response whose generation failed upstream (finish_reason "error")."""


def _read(req: urllib.request.Request, deadline: float) -> bytes:
    """POST and read the body under a wall-clock deadline. Some gateways keep slow requests alive by
    streaming whitespace, so a per-read socket timeout alone can wait forever on a stalled upstream."""
    end = time.monotonic() + deadline
    with urllib.request.urlopen(req, timeout=min(deadline, 300)) as r:
        body = b""
        while chunk := r.read1(65536):
            body += chunk
            if time.monotonic() > end:
                raise TimeoutError(f"no complete response within {deadline}s")
        return body


def _rejected_param(error_body: bytes) -> str | None:
    try:
        return json.loads(error_body)["error"].get("param")
    except (ValueError, KeyError, AttributeError, TypeError):
        return None


def pmap(fn, items, workers: int = 16, desc: str = "") -> list:
    items, out, done = list(items), [], 0
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(fn, items):
            out.append(res)
            done += 1
            if done % max(1, len(items) // 20) == 0 or done == len(items):
                print(f"[{desc}] {done}/{len(items)}", flush=True)
    return out


def import_original(repo: str | Path, module: str, stub: tuple[str, ...] = ()):
    """Import `module` from an original repo checkout. `stub` names heavy imports the code path we use
    never touches (e.g. vllm, torch); they are replaced by empty modules instead of being installed."""
    for name in stub:
        mod = types.ModuleType(name)
        mod.__getattr__ = lambda attr, _n=name: type(attr, (), {"__module__": _n})
        sys.modules.setdefault(name, mod)
    sys.path.insert(0, str(repo))
    return importlib.import_module(module)


def mean(xs) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]  # bools count as 0/1
    return sum(xs) / len(xs) if xs else None


def write_results(metrics: dict, breakdown: dict | None = None, n: int | None = None,
                  samples: list | None = None) -> None:
    """Write results.json in the shared schema (+ samples.jsonl) to MHEVAL_OUTPUT_DIR."""
    out = Path(os.environ["MHEVAL_OUTPUT_DIR"])
    (out / "results.json").write_text(json.dumps({"metrics": metrics, "breakdown": breakdown or {}, "n": n}, indent=2))
    if samples is not None:
        with open(out / "samples.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))
