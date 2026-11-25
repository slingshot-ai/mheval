import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from mheval.config import load_tasks

requests = pytest.importorskip("requests")

REPO_ROOT = Path(__file__).resolve().parents[1]


class StrictJudge(BaseHTTPRequestHandler):
    """Rejects `max_tokens` and `temperature` for the model `strict`, as GPT-5.6 Sol rejects `max_tokens`."""

    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        if body["model"] == "strict" and ("max_tokens" in body or "temperature" in body):
            param = "max_tokens" if "max_tokens" in body else "temperature"
            return self._send(400, {"error": {"message": f"Unsupported parameter: '{param}'", "param": param}})
        self._send(200, {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def task():
    return load_tasks()["spiral_bench"]


@pytest.fixture
def judge_url():
    StrictJudge.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), StrictJudge)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"
    httpd.shutdown()


@pytest.fixture
def adapted(judge_url, monkeypatch):
    monkeypatch.setattr(requests.Session, "request", requests.Session.request)  # undone after the test
    spec = importlib.util.spec_from_file_location("spiral_judge_requests", task().task_dir / "judge_requests.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install(judge_url, {"strict"})
    return judge_url


def test_default_judge_is_gpt_5_6_sol():
    assert task().roles["judge"]["model"] == "gpt-5.6-sol"


def test_when_the_judge_rejects_params_then_they_are_adapted_once(adapted):
    for _ in range(2):
        body = {"model": "strict", "messages": [], "max_tokens": 8096, "temperature": 0.3}
        assert requests.post(adapted, json=body).status_code == 200
    assert StrictJudge.requests == [
        {"model": "strict", "messages": [], "max_tokens": 8096, "temperature": 0.3},
        {"model": "strict", "messages": [], "temperature": 0.3, "max_completion_tokens": 8096},
        {"model": "strict", "messages": [], "max_completion_tokens": 8096},
        {"model": "strict", "messages": [], "max_completion_tokens": 8096},
    ]


def test_when_another_model_uses_the_judge_endpoint_then_its_requests_pass_through(adapted):
    body = {"model": "target", "messages": [], "max_tokens": 3072, "temperature": 0.3}
    requests.post(adapted, json=body)
    assert StrictJudge.requests == [body]


def test_when_run_as_the_task_command_then_the_script_imports_from_its_checkout_and_its_judge_is_adapted(
    judge_url, tmp_path
):
    (tmp_path / "api_client.py").write_text(
        "import requests\n"
        "def judge(url):\n"
        "    return requests.post(url, json={'model': 'strict', 'messages': [], 'max_tokens': 8096}).status_code\n"
    )
    (tmp_path / "main.py").write_text("import os, api_client\nprint(api_client.judge(os.environ['JUDGE_BASE_URL']))\n")
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "JUDGE_BASE_URL": judge_url,
        "JUDGE_MODEL": "strict",
        "TARGET_BASE_URL": "http://127.0.0.1:9/v1",
        "TARGET_MODEL": "target",
    }
    run = subprocess.run(
        [sys.executable, str(task().task_dir / "judge_requests.py"), "main.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert run.stdout.strip().splitlines()[-1] == "200"
    assert StrictJudge.requests[-1] == {"model": "strict", "messages": [], "max_completion_tokens": 8096}
