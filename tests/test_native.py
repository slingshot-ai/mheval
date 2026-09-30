import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import pytest

from mheval.native import Chat, mean


class FakeReasoningModel(BaseHTTPRequestHandler):
    """Behaves like an OpenAI reasoning model: rejects temperature and max_tokens, and spends small
    completion budgets entirely on hidden reasoning."""

    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        for param in ("temperature", "max_tokens"):
            if param in body:
                return self._send(400, {"error": {"message": f"Unsupported parameter: '{param}'", "param": param}})
        if body.get("max_completion_tokens", 10**6) < 5000:  # reasoning uses almost all of it: a truncated answer
            return self._send(200, {"choices": [{"message": {"content": "The right ther"}, "finish_reason": "length"}],
                                    "usage": {"completion_tokens_details": {"reasoning_tokens": 1000}}})
        return self._send(200, {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    httpd = HTTPServer(("127.0.0.1", 0), FakeReasoningModel)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    FakeReasoningModel.requests = []
    monkeypatch.setenv("TARGET_MODEL", "fake")
    monkeypatch.setenv("TARGET_BASE_URL", f"http://127.0.0.1:{httpd.server_port}/v1")
    monkeypatch.setenv("TARGET_API_KEY", "k")
    monkeypatch.setenv("TARGET_PARAMS", json.dumps({"temperature": 0.7}))
    yield
    httpd.shutdown()


def test_chat_adapts_to_reasoning_model(server):
    chat = Chat("target")
    assert chat("hi", max_tokens=1024) == "answer"
    assert chat.rejected == {"temperature", "max_tokens"}
    assert chat.headroom == Chat.REASONING_HEADROOM
    final = FakeReasoningModel.requests[-1]
    assert "temperature" not in final and final["max_completion_tokens"] == 1024 + Chat.REASONING_HEADROOM
    # later calls reuse what was learned: a single request
    n = len(FakeReasoningModel.requests)
    assert chat("again", max_tokens=1024) == "answer"
    assert len(FakeReasoningModel.requests) == n + 1


def test_mean():
    assert mean([1, 2, None, "x"]) == 1.5
    assert mean([True, False]) == 0.5
    assert mean([]) is None


class Trickle(BaseHTTPRequestHandler):
    """Keeps the connection alive with whitespace and never finishes the body."""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                self.wfile.write(b"1\r\n \r\n")
                self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def test_chat_deadline_on_trickling_response(monkeypatch):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("TARGET_MODEL", "fake")
    monkeypatch.setenv("TARGET_BASE_URL", f"http://127.0.0.1:{httpd.server_port}/v1")
    monkeypatch.setenv("TARGET_PARAMS", "{}")
    monkeypatch.setattr(Chat, "DEADLINE", 0.5)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        Chat("target")("hi", retries=2)
    assert time.monotonic() - start < 10
    httpd.shutdown()


class FlakyEmpty(BaseHTTPRequestHandler):
    """Returns an empty `stop` answer and a provider error before a real answer."""

    replies = []

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        choice = type(self).replies.pop(0)
        data = json.dumps({"choices": [choice]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_chat_retries_empty_and_provider_errors(monkeypatch):
    FlakyEmpty.replies = [{"message": {"content": ""}, "finish_reason": "stop"},
                          {"message": {"content": ""}, "finish_reason": "error", "error": {"code": 502}},
                          {"message": {"content": "answer"}, "finish_reason": "stop"}]
    httpd = HTTPServer(("127.0.0.1", 0), FlakyEmpty)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("TARGET_MODEL", "fake")
    monkeypatch.setenv("TARGET_BASE_URL", f"http://127.0.0.1:{httpd.server_port}/v1")
    monkeypatch.setenv("TARGET_PARAMS", "{}")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert Chat("target")("hi") == "answer"
    httpd.shutdown()
