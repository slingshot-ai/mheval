"""Run a benchmark's own script, adapting its requests for the model under test to reasoning models.

    python -m mheval.reasoning_models main.py <args>

Some benchmarks call the API with their own client, written for the models of their day. For the model under test,
this applies the adaptations `native.Chat` makes in mheval's own adapters: a `temperature`/`top_p` the endpoint
rejects is dropped and a rejected `max_tokens` is sent as `max_completion_tokens` (both learned once), and a reply
cut off at the token cap by hidden reasoning, with no answer, is retried once with extra room. Every other request
(judges, simulated users) passes through unchanged. Stdlib only, plus the `requests` the benchmark itself uses.
"""

from __future__ import annotations

import os
import runpy
import sys

from mheval.native import Chat, _rejected_param


def install() -> None:
    import requests

    endpoint = os.environ["TARGET_BASE_URL"].rstrip("/") + "/chat/completions"
    model = os.environ["TARGET_MODEL"]
    send = requests.Session.request
    rejected: set[str] = set()

    def log(msg: str) -> None:
        print(f"[mheval] {model}: {msg}", flush=True)

    def adapt(payload: dict) -> dict:
        payload = dict(payload)
        for param in rejected:
            value = payload.pop(param, None)
            if param == "max_tokens" and value is not None:
                payload["max_completion_tokens"] = value
        return payload

    def request(self, method, url, *args, **kwargs):  # the names requests passes them by
        payload = kwargs.get("json")
        if (
            method.upper() != "POST"
            or url.rstrip("/") != endpoint
            or not isinstance(payload, dict)
            or payload.get("model") != model
        ):
            return send(self, method, url, *args, **kwargs)
        kwargs["json"] = adapt(payload)
        response = send(self, method, url, *args, **kwargs)
        while response.status_code == 400:
            param = _rejected_param(response.content)
            if param not in ("temperature", "top_p", "max_tokens") or param not in kwargs["json"]:
                break  # not an adaptable rejection, or this request already left the param out
            if param not in rejected:  # another thread may have learned it while this request was in flight
                rejected.add(param)
                log(f"endpoint rejected `{param}`; using the API default")
            kwargs["json"] = adapt(payload)
            response = send(self, method, url, *args, **kwargs)
        if not response.ok:
            return response
        try:
            choice = response.json()["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError):
            return response
        sent = kwargs["json"]
        cap = next((k for k in ("max_completion_tokens", "max_tokens") if k in sent), None)
        if cap and choice.get("finish_reason") == "length" and not (choice.get("message") or {}).get("content"):
            log(f"answer cut off at the token cap by hidden reasoning; adding {Chat.REASONING_HEADROOM} tokens")
            kwargs["json"] = {**sent, cap: sent[cap] + Chat.REASONING_HEADROOM}
            return send(self, method, url, *args, **kwargs)
        return response

    requests.Session.request = request


if __name__ == "__main__":
    install()
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
