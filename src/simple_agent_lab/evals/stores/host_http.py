"""Host-run HTTP artifact store: batteries-included, no third-party middleware.

For a remote Docker daemon (or any container that cannot share the host
filesystem) but where you do not want to stand up S3, the host runs a tiny
stdlib `http.server` over the run directory. The container reads inputs and
pushes outputs + live trace to it over HTTP — so it is "bind mount over the
network." Zero external dependencies; `uv sync` is all you need.

Two classes, both `ArtifactStore`:

- `HostHttpStore` — host side. A context manager that serves a base directory
  and hands the container a URL via `container_binding`. Host `get`/`put` are
  plain disk ops (the host has the disk); the server persists container writes
  to that same disk, so reading results afterwards is a normal file read.
- `HttpArtifactClient` — container side. `get`/`put` over HTTP. Reconstructed
  from env by `container_store_from_env`.

The endpoint is bound to localhost with a random token by default; it accepts
writes, so do not expose it publicly. Reaching it from a local container uses
``host.docker.internal`` (mapped to ``host-gateway`` on Linux via
`ContainerBinding.add_hosts`).
"""

from __future__ import annotations

import secrets
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..protocols import ContainerBinding

TOKEN_HEADER = "X-SAL-Token"
DEFAULT_CONTAINER_HOST = "host.docker.internal"


def _safe_join(base: Path, rel: str) -> Path:
    target = (base / rel.lstrip("/")).resolve()
    base_resolved = base.resolve()
    if base_resolved != target and base_resolved not in target.parents:
        raise PermissionError(f"path escapes store root: {rel!r}")
    return target


class HostHttpStore:
    """Serve a base directory over HTTP; hand the container a URL to push to."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        bind_host: str = "127.0.0.1",
        container_host: str = DEFAULT_CONTAINER_HOST,
        token: str | None = None,
        rel: Path | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.bind_host = bind_host
        self.container_host = container_host
        self.token = token or secrets.token_urlsafe(16)
        self._rel = rel if rel is not None else Path(".")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle (host side) --------------------------------------------- #
    def __enter__(self) -> "HostHttpStore":
        self.base_dir.mkdir(parents=True, exist_ok=True)
        handler = _make_handler(self.base_dir, self.token)
        self._server = ThreadingHTTPServer((self.bind_host, 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._server = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("HostHttpStore must be entered before use")
        return self._server.server_address[1]

    # -- ArtifactStore ----------------------------------------------------- #
    def bind(self, run_dir: Path) -> "HostHttpStore":
        rel = Path(run_dir).resolve().relative_to(self.base_dir.resolve())
        view = HostHttpStore(
            self.base_dir,
            bind_host=self.bind_host,
            container_host=self.container_host,
            token=self.token,
            rel=rel,
        )
        # Share the running server with the bound view.
        view._server = self._server
        view._thread = self._thread
        return view

    def _path(self, key: str) -> Path:
        return _safe_join(self.base_dir, (self._rel / key).as_posix())

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def container_binding(self) -> ContainerBinding:
        url = f"http://{self.container_host}:{self.port}/{self._rel.as_posix()}"
        return ContainerBinding(
            env={
                "SAL_STORE": "http",
                "SAL_STORE_URL": url.rstrip("/"),
                "SAL_STORE_TOKEN": self.token,
            },
            add_hosts={DEFAULT_CONTAINER_HOST: "host-gateway"},
        )

    def collect_outputs(self) -> None:
        # The server already persisted container writes to base_dir on disk.
        return None


class HttpArtifactClient:
    """Container-side `ArtifactStore`: `get`/`put` over HTTP to a host store."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def bind(self, run_dir: Path) -> "HttpArtifactClient":
        return self

    def _url(self, key: str) -> str:
        return f"{self.base_url}/{key.lstrip('/')}"

    def get(self, key: str) -> bytes:
        req = urllib.request.Request(self._url(key), headers={TOKEN_HEADER: self.token})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def put(self, key: str, data: bytes) -> None:
        req = urllib.request.Request(
            self._url(key),
            data=data,
            method="PUT",
            headers={TOKEN_HEADER: self.token},
        )
        urllib.request.urlopen(req, timeout=30).close()

    def exists(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except urllib.error.HTTPError:
            return False

    def container_binding(self) -> ContainerBinding:  # pragma: no cover - host-only
        raise NotImplementedError("HttpArtifactClient is the container side")

    def collect_outputs(self) -> None:  # pragma: no cover - host-only
        return None


def _make_handler(base_dir: Path, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # silence logging
            return None

        def _check_token(self) -> bool:
            if self.headers.get(TOKEN_HEADER) == token:
                return True
            self.send_error(HTTPStatus.FORBIDDEN, "bad token")
            return False

        def do_GET(self) -> None:
            if not self._check_token():
                return
            try:
                data = _safe_join(base_dir, self.path).read_bytes()
            except (FileNotFoundError, PermissionError):
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_PUT(self) -> None:
            if not self._check_token():
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                path = _safe_join(base_dir, self.path)
            except PermissionError:
                self.send_error(HTTPStatus.FORBIDDEN, "path escapes root")
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

    return Handler
