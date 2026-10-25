"""Shared JSON and static-file server plumbing for local game applications."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class JSONAppHandler(SimpleHTTPRequestHandler):
    web_root: Path
    index_file: str
    model_lock = threading.Lock()

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(self.web_root), **kwargs)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 100_000:
            raise ValueError("Request is too large")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/":
            self.path = f"/{self.index_file}"
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} {format % args}")


def serve_app(
    handler: type[JSONAppHandler], host: str, port: int, label: str
) -> None:
    handler.model_lock = threading.Lock()
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving {label} at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
