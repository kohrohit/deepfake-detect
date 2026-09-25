"""HTTP surface: six JSON endpoints and one page, on the standard library.

Bound to 127.0.0.1 by default. There is no authentication, which is a
deliberate scope decision recorded here rather than forgotten: this service
answers "what did the detector say about this file", and the honest answer
today is `insufficient_evidence` (see /api/evidence). Do NOT expose the port
beyond the host without putting an authenticating reverse proxy in front —
`POST /api/scan` writes attacker-chosen bytes to the workdir and spends CPU
decoding them.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from ..errors import DfdError
from ..limits import DEFAULT_LIMITS
from .dashboard import DASHBOARD_HTML
from .store import QUEUED, Store
from .worker import Worker

logger = logging.getLogger(__name__)

#: Upload ceiling for the HTTP path. Same value the decode limits use, so a
#: file that would be refused at decode is refused at the door instead of
#: after being written to disk.
DEFAULT_MAX_UPLOAD_BYTES = DEFAULT_LIMITS.max_file_bytes

#: Seconds a request may hold a connection without progress. Without this a
#: client that sends `Content-Length: 268435456` and then stops writing holds
#: a worker thread until the process dies — the classic slow-read denial of
#: service, and free to mount because the read is what this service does
#: before it knows anything about the caller.
DEFAULT_REQUEST_TIMEOUT_S = 30.0

#: Concurrent requests served before further ones are refused with 503.
#: `ThreadingHTTPServer` spawns one thread per connection with no ceiling, so
#: without this the bound is the OS thread limit and the failure mode is the
#: whole process rather than one rejected request. Scoring happens on the
#: worker, not here, so this only has to cover JSON reads and uploads.
DEFAULT_MAX_CONCURRENT_REQUESTS = 32

#: Largest `limit` an API listing will honour.
MAX_LISTING_LIMIT = 500


class _Handler(BaseHTTPRequestHandler):
    server_version = "dfd"
    # Suppress the default "protocol version" banner leaking a Python version.
    sys_version = ""
    #: `StreamRequestHandler.setup` applies this to the connection socket, so
    #: it bounds every read this handler performs. See
    #: DEFAULT_REQUEST_TIMEOUT_S for why a service with an unbounded read is
    #: a service with an unbounded thread count.
    timeout = DEFAULT_REQUEST_TIMEOUT_S

    # Set by make_server on the server object; read through self.server.
    @property
    def _store(self) -> Store:
        return cast(Store, self.server.store)  # type: ignore[attr-defined]

    @property
    def _worker(self) -> Worker:
        return cast(Worker, self.server.worker)  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # BaseHTTPRequestHandler writes to stderr directly; route it through
        # logging so the service has one log stream, not two.
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: Any, *,
              content_type: str = "application/json") -> None:
        body = (payload if isinstance(payload, bytes)
                else json.dumps(payload, allow_nan=False).encode())
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # This service renders a dashboard that fetches its own JSON; nothing
        # here is meant to be embedded elsewhere.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": message})

    def _slot(self) -> bool:
        """Take a concurrency slot, or refuse the request with 503.

        Refusing is the point. An accepted request that cannot be served is
        indistinguishable to the caller from a service that is down, except
        that it also costs a thread.
        """
        sem: threading.Semaphore = self.server.request_slots  # type: ignore[attr-defined]
        if sem.acquire(blocking=False):
            return True
        logger.warning("refusing request: %d concurrent slots all taken",
                       self.server.max_concurrent)  # type: ignore[attr-defined]
        self._error(HTTPStatus.SERVICE_UNAVAILABLE,
                    "too many concurrent requests; retry")
        return False

    def _release(self) -> None:
        sem: threading.Semaphore = self.server.request_slots  # type: ignore[attr-defined]
        sem.release()

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if not self._slot():
            return
        try:
            self._get()
        finally:
            self._release()

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if not self._slot():
            return
        try:
            self._post()
        finally:
            self._release()

    def _get(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        if route == "/":
            self._send(HTTPStatus.OK, DASHBOARD_HTML.encode(),
                       content_type="text/html; charset=utf-8")
            return
        if route == "/health":
            self._send(HTTPStatus.OK, self._health())
            return
        if route == "/api/stats":
            self._send(HTTPStatus.OK, self._store.stats())
            return
        if route == "/api/evidence":
            self._send(HTTPStatus.OK, self._evidence())
            return
        if route == "/api/submissions":
            try:
                limit = _listing_limit(query.get("limit", ["50"])[0])
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            status = query.get("status", [None])[0]
            rows = self._store.recent(limit=limit, status=status)
            self._send(HTTPStatus.OK,
                       {"submissions": [r.summary() for r in rows]})
            return
        if route.startswith("/api/submissions/"):
            sid = route.rsplit("/", 1)[-1]
            row = self._store.get(sid)
            if row is None:
                self._error(HTTPStatus.NOT_FOUND, f"no submission {sid!r}")
                return
            payload = row.summary()
            # Whether the INPUT is still on disk. Retention deletes scored
            # files after a window while keeping every record, so a reader
            # looking at an old decision must be able to tell "the file is
            # gone" from "the path is wrong".
            payload["file_retained"] = Path(row.path).is_file()
            # The full record is attached HERE and never in the listing: it
            # is the thing a reader actually needs when looking at one
            # decision, and the thing that makes a listing unusable.
            payload["record"] = (json.loads(row.record_json)
                                 if row.record_json else None)
            self._send(HTTPStatus.OK, payload)
            return
        self._error(HTTPStatus.NOT_FOUND, f"no route {route!r}")

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self.do_GET()

    def _post(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/api/scan":
            self._error(HTTPStatus.NOT_FOUND, f"no route {parsed.path!r}")
            return

        # This handler reads exactly `Content-Length` bytes and does not
        # decode chunked transfer encoding. Left unchecked, a chunked upload
        # arrives as length 0 and is accepted as an EMPTY submission, which
        # then fails at decode and is recorded as a refusal about the file
        # rather than about the request. Refuse it at the door instead.
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self._error(HTTPStatus.LENGTH_REQUIRED,
                        "chunked transfer encoding is not supported; send "
                        "Content-Length")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "malformed Content-Length")
            return
        if length < 0:
            self._error(HTTPStatus.BAD_REQUEST, "negative Content-Length")
            return
        limit = self.server.max_upload_bytes  # type: ignore[attr-defined]
        if length > limit:
            # Refused from the header, before reading the body: reading it to
            # find out how big it is, is the denial of service.
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"upload of {length} bytes exceeds the {limit}-byte limit")
            return
        data = self.rfile.read(length) if length else b""
        if len(data) != length:
            # The socket closed or timed out mid-body. Accepting the prefix
            # would store a truncated file and report a decode failure about
            # content the caller never finished sending.
            self._error(HTTPStatus.BAD_REQUEST,
                        f"body ended after {len(data)} of {length} bytes")
            return

        query = parse_qs(parsed.query)
        filename = Path(query.get("filename", ["upload"])[0]).name or "upload"
        try:
            sid = self._worker.submit_bytes(data, filename=filename,
                                            source="api")
        except DfdError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send(HTTPStatus.ACCEPTED, {"id": sid, "status": QUEUED})

    def _health(self) -> dict[str, Any]:
        stats = self._store.stats()
        started: float = self.server.started_at  # type: ignore[attr-defined]
        return {
            "status": "ok",
            "queue_depth": stats["status"].get(QUEUED, 0),
            "uptime_s": round(time.time() - started, 1),
            "submissions": sum(stats["status"].values()),
        }

    def _evidence(self) -> dict[str, Any]:
        """The measured detector report card, or an explicit 'not measured'.

        Never a 404. The dashboard reads this to decide whether to show a
        verdict plainly or behind a warning, and a missing file must produce
        the warning, not a silent fetch failure that renders nothing.
        """
        path: Path = self.server.evidence_path  # type: ignore[attr-defined]
        if not path.is_file():
            return {"available": False,
                    "note": "No measured detector performance on this "
                            "deployment. Every verdict below is unvalidated."}
        try:
            data = cast("dict[str, Any]", json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            logger.warning("unreadable evidence card %s: %s", path, exc)
            return {"available": False, "note": f"unreadable report card: {exc}"}
        data["available"] = True
        return data


def _listing_limit(raw: str) -> int:
    """Parse and bound a listing `limit`, or say why it is unacceptable.

    Two real defects, both fixed here rather than downstream:

    - A non-integer (`?limit=abc`) raised inside the handler, which sends no
      response at all: the caller sees a closed connection, not a 400.
    - A NEGATIVE value reached SQLite, where `LIMIT -1` means *no limit* —
      so `?limit=-1` returned the entire submissions table from an endpoint
      whose whole contract is that it is bounded. Verified against sqlite3
      2026-09-23: `LIMIT -1` yields every row, `LIMIT 0` yields none.
    """
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"limit must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"limit must be at least 1, got {value}")
    return min(value, MAX_LISTING_LIMIT)


def make_server(host: str, port: int, *, store: Store, worker: Worker,
                evidence_path: str | Path,
                max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
                max_concurrent: int = DEFAULT_MAX_CONCURRENT_REQUESTS
                ) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server.

    Args:
        host: interface to bind. Default deployments bind 127.0.0.1.
        port: TCP port; 0 asks the OS for a free one, which is what the
            tests use so they never collide with a running service.
        store: submission store, read by the JSON endpoints.
        worker: used only for `submit_bytes`; the server never scores.
        evidence_path: JSON report card of measured detector performance.
        max_upload_bytes: refuse a larger POST from its Content-Length.
        max_concurrent: requests served at once; further ones get 503 rather
            than a thread each.

    Returns:
        A `ThreadingHTTPServer`. Call `serve_forever` on it, and `shutdown`
        then `server_close` to stop.
    """
    httpd = ThreadingHTTPServer((host, port), _Handler)
    # Threads are daemons so a stuck request cannot keep the process alive
    # after the service has been told to stop.
    httpd.daemon_threads = True
    httpd.store = store  # type: ignore[attr-defined]
    httpd.worker = worker  # type: ignore[attr-defined]
    httpd.evidence_path = Path(evidence_path)  # type: ignore[attr-defined]
    httpd.max_upload_bytes = int(max_upload_bytes)  # type: ignore[attr-defined]
    httpd.max_concurrent = int(max_concurrent)  # type: ignore[attr-defined]
    httpd.request_slots = threading.Semaphore(  # type: ignore[attr-defined]
        int(max_concurrent))
    httpd.started_at = time.time()  # type: ignore[attr-defined]
    return httpd
