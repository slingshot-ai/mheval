"""Run Spiral-Bench's own script with its judge requests adapted to the judge.

    python judge_requests.py main.py <args>

Spiral-Bench's client sends every judge `max_tokens` and `temperature: 0.3`, and adapts them only for the judges it
names (GPT-5, o3). Requests for a judge (`JUDGE_MODEL`, a comma list) to its endpoint (`JUDGE_BASE_URL`) get the
adaptation `mheval.reasoning_models` makes for the model under test: a `temperature`/`top_p` the endpoint rejects is
dropped and a rejected `max_tokens` is sent as `max_completion_tokens`, each learned once. The script then runs under
`mheval.reasoning_models`. Stdlib only, plus the `requests` the benchmark itself uses.
"""

from __future__ import annotations

import os
import runpy
import sys

ADAPTABLE = ("temperature", "top_p", "max_tokens")


def install(endpoint: str, judges: set[str]) -> None:
    import requests

    from mheval.native import _rejected_param

    send = requests.Session.request
    rejected: set[str] = set()

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
            or payload.get("model") not in judges
        ):
            return send(self, method, url, *args, **kwargs)
        kwargs["json"] = adapt(payload)
        response = send(self, method, url, *args, **kwargs)
        while response.status_code == 400:
            param = _rejected_param(response.content)
            if param not in ADAPTABLE or param not in kwargs["json"]:
                break  # not an adaptable rejection, or this request already left the param out
            if param not in rejected:  # another thread may have learned it while this request was in flight
                rejected.add(param)
                print(f"[mheval] judge {payload.get('model')}: endpoint rejected `{param}`; adapting", flush=True)
            kwargs["json"] = adapt(payload)
            response = send(self, method, url, *args, **kwargs)
        return response

    requests.Session.request = request


if __name__ == "__main__":
    from mheval import reasoning_models

    install(os.environ["JUDGE_BASE_URL"].rstrip("/"), set(os.environ["JUDGE_MODEL"].split(",")))
    reasoning_models.install()
    sys.argv = sys.argv[1:]
    sys.path[0] = os.getcwd()  # the benchmark's checkout, as `python -m` would have it, not this task folder
    runpy.run_path(sys.argv[0], run_name="__main__")
