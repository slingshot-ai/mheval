"""Forward OpenAI-compatible requests to a Google endpoint (Vertex AI) with a fresh Google access token.

    python google_auth_proxy.py <upstream base URL> <port file>

EQ-Bench 4 reads each API key once, but a Google access token lasts an hour and a run takes several. This listens on
a free localhost port (written to <port file>), replaces each request's Authorization header with a current token
from the application default credentials, and passes the request and its response through unchanged.
"""

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import google.auth
import google.auth.transport.requests
import requests

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
TIMEOUT = 900


class Token:
    def __init__(self):
        self.credentials, _ = google.auth.default(scopes=SCOPES)
        self.lock = threading.Lock()

    def get(self) -> str:
        with self.lock:
            if not self.credentials.valid:
                self.credentials.refresh(google.auth.transport.requests.Request())
            return self.credentials.token


def serve(upstream: str, port_file: Path) -> None:
    token = Token()
    token.get()  # fail now, not on the first persona turn, if there are no usable credentials

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            headers = {"Authorization": f"Bearer {token.get()}", "Content-Type": "application/json"}
            try:
                r = requests.post(upstream.rstrip("/") + self.path, data=body, headers=headers, timeout=TIMEOUT)
                status, content, ctype = r.status_code, r.content, r.headers.get("Content-Type", "application/json")
            except requests.RequestException as e:
                status, content, ctype = 502, str(e).encode(), "text/plain"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port_file.write_text(str(server.server_address[1]))
    server.serve_forever()


if __name__ == "__main__":
    serve(sys.argv[1], Path(sys.argv[2]))
