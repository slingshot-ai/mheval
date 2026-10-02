import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

requests = pytest.importorskip("requests")


class FakeReasoningModel(BaseHTTPRequestHandler):
    """Spends any cap under 5000 tokens on hidden reasoning and returns no answer."""

    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        if body["model"] == "strict" and ("max_tokens" in body or "temperature" in body):
            param = "max_tokens" if "max_tokens" in body else "temperature"
            return self._send(400, {"error": {"message": f"Unsupported parameter: '{param}'", "param": param}})
        if body.get("max_tokens", body.get("max_completion_tokens", 10**6)) < 5000:
            choice = {"message": {"content": None}, "finish_reason": "length"}
        else:
            choice = {"message": {"content": "answer"}, "finish_reason": "stop"}
        self._send(200, {"choices": [choice]})

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def url(monkeypatch, request):
    FakeReasoningModel.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), FakeReasoningModel)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}/v1"
    monkeypatch.setenv("TARGET_BASE_URL", base)
    monkeypatch.setenv("TARGET_MODEL", getattr(request, "param", "target"))
    monkeypatch.setattr(requests.Session, "request", requests.Session.request)  # undone after the test
    from mheval.reasoning_models import install

    install()
    yield f"{base}/chat/completions"
    httpd.shutdown()


def test_target_cut_off_by_reasoning_is_retried_with_room(url):
    reply = requests.post(url, json={"model": "target", "messages": [], "max_tokens": 3072}).json()
    assert reply["choices"][0]["message"]["content"] == "answer"
    assert [r["max_tokens"] for r in FakeReasoningModel.requests] == [3072, 3072 + 16000]


def test_other_models_pass_through(url):
    reply = requests.post(url, json={"model": "judge", "messages": [], "max_tokens": 3072}).json()
    assert reply["choices"][0]["message"]["content"] is None
    assert len(FakeReasoningModel.requests) == 1


@pytest.mark.parametrize("url", ["strict"], indirect=True)
def test_rejected_params_are_adapted_once_like_native_chat(url):
    for _ in range(2):
        payload = {"model": "strict", "messages": [], "max_tokens": 9000, "temperature": 0.7}
        reply = requests.post(url, json=payload).json()
        assert reply["choices"][0]["message"]["content"] == "answer"
    sent = FakeReasoningModel.requests
    assert sent[-1] == {"model": "strict", "messages": [], "max_completion_tokens": 9000}
    assert len(sent) == 4  # two rejections on the first call, none after


@pytest.mark.parametrize("url", ["strict"], indirect=True)
def test_concurrent_first_requests_all_get_an_answer(url):
    from concurrent.futures import ThreadPoolExecutor

    payload = {"model": "strict", "messages": [], "max_tokens": 9000, "temperature": 0.7}
    with ThreadPoolExecutor(8) as pool:
        replies = list(pool.map(lambda _: requests.post(url, json=payload), range(16)))
    assert all(r.ok and r.json()["choices"][0]["message"]["content"] == "answer" for r in replies)
